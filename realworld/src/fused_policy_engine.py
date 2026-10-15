#!/usr/bin/env python3
"""
realworld/src/fused_policy_engine.py — Zero-PyTorch-Eager Fused Policy Engine for Robot CPU.
========================================================================================
Runs the fused Image -> DINOv3 (L2 Normalized) -> Actor graph directly using either:
  1. Intel OpenVINO (preferred for Intel CPU / iGPU onboard computers)
  2. ONNX Runtime (CPUExecutionProvider fallback)

Eliminates timm, PyTorch eager execution overhead, and Python GIL contention during
real-time closed-loop 10Hz navigation.

Features:
  - Direct NumPy / OpenCV zero-copy interfacing.
  - Native 4-frame temporal history FIFO management.
  - Automatic OpenVINO vs ONNX Runtime auto-detection.
  - Low-latency Intel CPU performance hints and threading configuration.
"""

import sys
import logging
import time
from pathlib import Path
from collections import deque
from typing import Tuple, Optional, Union

import numpy as np
from PIL import Image as PILImage

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger("fused_policy_engine")


class FusedPolicyEngine:
    """
    High-performance, zero-eager-PyTorch inference engine for MINav policy.
    Supports OpenVINO IR (.xml) and ONNX (.onnx).
    """

    IMAGE_H = 448
    IMAGE_W = 784
    PHI_DIM = 384
    STACK_FRAMES = 4

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "CPU",
        num_threads: int = 4,
        backend: str = "auto",  # 'auto', 'openvino', or 'onnx'
    ):
        self.model_path = Path(model_path).resolve()
        self.device = device.upper()
        self.num_threads = num_threads
        self.backend_type = None

        # FIFO buffer of normalized phi embeddings: stores up to (STACK_FRAMES - 1) past embeddings
        self.phi_history = deque(maxlen=self.STACK_FRAMES - 1)
        self.current_goal_phi: Optional[np.ndarray] = None  # [1, 384]

        # Resolve backend
        if backend == "auto":
            # Prefer OpenVINO if .xml exists or if given .onnx and openvino is installed
            xml_candidate = self.model_path.with_suffix(".xml")
            if self.model_path.suffix == ".xml" or xml_candidate.exists():
                self.backend_type = "openvino"
                self.target_path = xml_candidate if xml_candidate.exists() else self.model_path
            else:
                self.backend_type = "onnx"
                self.target_path = self.model_path
        else:
            self.backend_type = backend.lower()
            self.target_path = self.model_path

        # Initialize chosen backend
        if self.backend_type == "openvino":
            self._init_openvino()
        elif self.backend_type == "onnx":
            self._init_onnx()
        else:
            raise ValueError(f"Unknown backend type: {backend}")

        logger.info(
            f"FusedPolicyEngine initialized: Backend={self.backend_type.upper()} | "
            f"Device={self.device} | Threads={self.num_threads} | Model={self.target_path.name}"
        )

    def _init_openvino(self):
        try:
            import openvino as ov
        except ImportError:
            logger.warning("OpenVINO requested or detected, but 'openvino' is not installed. Falling back to ONNX Runtime.")
            self.backend_type = "onnx"
            self.target_path = self.model_path.with_suffix(".onnx")
            self._init_onnx()
            return

        core = ov.Core()
        ov_model = core.read_model(str(self.target_path))
        
        # Configure Intel CPU latency profile
        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_NUM_THREADS": self.num_threads,
        }
        
        compiled_model = core.compile_model(ov_model, self.device, config)
        self.infer_request = compiled_model.create_infer_request()
        self._infer_fn = self._infer_openvino

    def _init_onnx(self):
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = self.num_threads
        sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers = ["CPUExecutionProvider"]
        if self.device == "GPU" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers.insert(0, "CUDAExecutionProvider")

        self.ort_session = ort.InferenceSession(str(self.target_path), sess_opts, providers=providers)
        self._infer_fn = self._infer_onnx

    def _preprocess_image(self, rgb: Union[np.ndarray, PILImage.Image]) -> np.ndarray:
        """
        Preprocess input camera frame to [1, 3, 448, 784] float32 in [0, 1].
        Matches ToTensor() without ImageNet normalization.
        """
        if isinstance(rgb, PILImage.Image):
            rgb = np.array(rgb)

        # Ensure RGB format
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        elif rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]

        # Fast Resize via OpenCV if available
        if cv2 is not None:
            if rgb.shape[0] != self.IMAGE_H or rgb.shape[1] != self.IMAGE_W:
                resized = cv2.resize(rgb, (self.IMAGE_W, self.IMAGE_H), interpolation=cv2.INTER_LINEAR)
            else:
                resized = rgb
        else:
            pil_img = PILImage.fromarray(rgb)
            resized = np.array(pil_img.resize((self.IMAGE_W, self.IMAGE_H), PILImage.BILINEAR))

        # HWC uint8 [0, 255] -> CHW float32 [0.0, 1.0]
        tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(tensor, axis=0)  # [1, 3, H, W]

    def set_goal(self, goal_input: Union[np.ndarray, str, Path]):
        """
        Set the navigation goal representation.
        Accepts either:
          - Precomputed L2-normalized phi vector: shape (384,) or (1, 384)
          - Goal RGB image (file path or numpy array)
        """
        if isinstance(goal_input, (str, Path)):
            path = Path(goal_input)
            img = PILImage.open(path).convert("RGB")
            goal_rgb = np.array(img)
            self.set_goal_from_image(goal_rgb)
            return

        arr = np.asarray(goal_input, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == self.PHI_DIM:
            arr = np.expand_dims(arr, axis=0)
        elif arr.ndim == 3:  # Image array
            self.set_goal_from_image(arr)
            return

        # Ensure unit L2 normalization
        norm = np.linalg.norm(arr, axis=-1, keepdims=True)
        if norm > 1e-6:
            arr = arr / norm
        self.current_goal_phi = arr
        logger.info(f"Goal embedding updated. Shape: {self.current_goal_phi.shape}, L2-norm: {np.linalg.norm(self.current_goal_phi):.3f}")

    def set_goal_from_image(self, goal_rgb: np.ndarray):
        """Encode goal RGB frame into unit normalized embedding."""
        img_tensor = self._preprocess_image(goal_rgb)
        dummy_past = np.zeros((1, 3 * self.PHI_DIM), dtype=np.float32)
        dummy_goal = np.zeros((1, self.PHI_DIM), dtype=np.float32)
        dummy_goal[0, 0] = 1.0

        # Run model to extract current_phi
        _, phi_encoded = self._infer_fn(img_tensor, dummy_past, dummy_goal)
        self.current_goal_phi = phi_encoded
        logger.info(f"Goal encoded from image. Shape: {self.current_goal_phi.shape}, L2-norm: {np.linalg.norm(self.current_goal_phi):.3f}")

    def reset(self):
        """Clear temporal history buffer."""
        self.phi_history.clear()

    def _infer_openvino(self, image: np.ndarray, past_phi: np.ndarray, goal_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inputs = {
            "current_image": image,
            "past_phi_stack": past_phi,
            "goal_phi": goal_phi,
        }
        self.infer_request.infer(inputs)
        action = self.infer_request.get_tensor("action").data
        current_phi = self.infer_request.get_tensor("current_phi").data
        return action, current_phi

    def _infer_onnx(self, image: np.ndarray, past_phi: np.ndarray, goal_phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        inputs = {
            "current_image": image,
            "past_phi_stack": past_phi,
            "goal_phi": goal_phi,
        }
        outs = self.ort_session.run(None, inputs)
        return outs[0], outs[1]

    def step(self, rgb_frame: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one control step from raw camera frame.

        Parameters:
          rgb_frame: np.ndarray of shape (H, W, 3) in RGB order.

        Returns:
          action: np.ndarray of shape (3,) with normalized [vx, vy, wz] in [-1, 1].
          similarity: float cosine similarity to active goal.
          is_ready: bool indicating whether the 4-frame temporal buffer is fully populated.
        """
        if self.current_goal_phi is None:
            raise RuntimeError("Goal not set! Call engine.set_goal(...) before step().")

        # 1. Preprocess camera frame
        img_tensor = self._preprocess_image(rgb_frame)

        # 2. Check buffer readiness
        is_ready = (len(self.phi_history) == self.STACK_FRAMES - 1)

        if not is_ready:
            # During warmup, repeat the earliest available phi or zero padding
            if len(self.phi_history) == 0:
                dummy_past = np.zeros((1, (self.STACK_FRAMES - 1) * self.PHI_DIM), dtype=np.float32)
            else:
                needed = (self.STACK_FRAMES - 1) - len(self.phi_history)
                padding = [self.phi_history[0]] * needed
                full_stack = padding + list(self.phi_history)
                dummy_past = np.concatenate(full_stack, axis=-1)
            past_phi_stack = dummy_past
        else:
            past_phi_stack = np.concatenate(list(self.phi_history), axis=-1)

        # 3. Fused inference pass
        action_out, current_phi = self._infer_fn(img_tensor, past_phi_stack, self.current_goal_phi)

        # 4. Slide temporal FIFO buffer
        self.phi_history.append(current_phi)

        # 5. Compute cosine similarity to goal: <phi_t, phi_g>
        similarity = float(np.sum(current_phi * self.current_goal_phi))

        normalized_action = action_out.flatten()
        return normalized_action, similarity, is_ready
