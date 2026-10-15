#!/usr/bin/env python3
"""
realworld/tools/export_fused_pipeline.py — Export Fused DINOv3 + Actor to ONNX and OpenVINO.
========================================================================================
Exports the MINav Image -> DINOv3 (L2 Normalized) -> Actor inference pipeline into
a single fused ONNX computational graph and optimized Intel OpenVINO IR (.xml / .bin).

Eliminates timm and eager PyTorch runtime overhead on the robot onboard CPU.

Supported Export Modes:
  1. 'step' (Default, recommended for 10Hz robot control):
     - Inputs:
         * current_image:  [1, 3, 448, 784] float32 in [0, 1]
         * past_phi_stack: [1, 1152]        float32 (phi_{t-3}, phi_{t-2}, phi_{t-1})
         * goal_phi:       [1, 384]         float32 (L2 normalized goal vector)
     - Outputs:
         * action:         [1, 3]           float32 normalized action in [-1, 1]
         * current_phi:    [1, 384]         float32 current frame L2 normalized embedding
  2. 'e2e' (Full 5-image end-to-end pipeline):
     - Inputs:
         * stacked_images: [1, 4, 3, 448, 784] float32 (4 consecutive camera frames)
         * goal_image:     [1, 3, 448, 784]    float32 (goal image frame)
     - Outputs:
         * action:         [1, 3]              float32 normalized action
         * similarity:     [1, 1]              float32 cosine similarity to goal
  3. 'encoder' (Standalone DINOv3 feature extractor):
     - Inputs:
         * image:          [B, 3, 448, 784]    float32
     - Outputs:
         * phi:            [B, 384]            float32 (L2 normalized CLS token)
  4. 'all': Exports all three models sequentially.

Usage:
  python realworld/tools/export_fused_pipeline.py --checkpoint models/run3/training/best_checkpoint.pt
  python realworld/tools/export_fused_pipeline.py --checkpoint models/run3/training/best_checkpoint.pt --mode all
"""

import sys
import os
import argparse
import logging
import time
from pathlib import Path

# Fix Windows console UTF-8 stream output
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# Project path resolution
ROOT = Path(__file__).resolve().parent.parent.parent
MUJOCO_ROOT = ROOT / "simulation" / "ranger_mini_v3"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MUJOCO_ROOT))

from src.rl.networks import Actor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("export_fused_pipeline")


def l2_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Universal L2 normalization avoiding ONNX ReduceL2 opset version conflicts."""
    return x / torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) + eps)


# ==============================================================================
# Fused Model Architectures
# ==============================================================================

class FusedStepPolicy(nn.Module):
    """
    Fused Step-wise Policy:
    Takes 1 new RGB image + 3 previous frame embeddings + 1 goal embedding.
    Computes DINOv3 feature extraction -> CLS token -> L2 norm -> Actor MLP.
    Returns normalized action [1, 3] and current frame embedding [1, 384].
    """
    def __init__(self, encoder: nn.Module, actor: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.actor = actor

    def forward(self, current_image: torch.Tensor, past_phi_stack: torch.Tensor, goal_phi: torch.Tensor):
        # 1. Feature extraction & L2 normalization
        feats = self.encoder.forward_features(current_image)
        cls_tok = feats[:, 0, :]
        current_phi = l2_norm(cls_tok)

        # 2. Chronological 4-frame state concatenation: [phi_{t-3}, phi_{t-2}, phi_{t-1}, phi_t]
        state = torch.cat([past_phi_stack, current_phi], dim=-1)

        # 3. Actor forward pass -> action in [-1, 1]^3
        action = self.actor(state, goal_phi)

        return action, current_phi


class FusedEndToEndPolicy(nn.Module):
    """
    Full End-to-End Multi-Frame Policy:
    Takes 4 consecutive camera frames [1, 4, 3, 448, 784] and 1 goal image [1, 3, 448, 784].
    Extracts all embeddings, computes cosine similarity, and evaluates Actor.
    """
    def __init__(self, encoder: nn.Module, actor: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.actor = actor

    def forward(self, stacked_images: torch.Tensor, goal_image: torch.Tensor):
        B = stacked_images.shape[0]
        # Reshape to batch encode all 4 frames: [B*4, 3, H, W]
        imgs_flat = stacked_images.reshape(B * 4, 3, 448, 784)
        feats_state = self.encoder.forward_features(imgs_flat)
        cls_state = feats_state[:, 0, :]
        phi_state = l2_norm(cls_state)                     # [B*4, 384]
        state = phi_state.reshape(B, 4 * 384)            # [B, 1536]

        # Encode goal image
        feats_goal = self.encoder.forward_features(goal_image)
        cls_goal = feats_goal[:, 0, :]
        goal_phi = l2_norm(cls_goal)                       # [B, 384]

        # Actor pass
        action = self.actor(state, goal_phi)

        # Cosine similarity between current frame (t0) and goal
        current_phi = phi_state.reshape(B, 4, 384)[:, 3, :]
        similarity = torch.sum(current_phi * goal_phi, dim=-1, keepdim=True)

        return action, similarity


class FusedDINOv3Encoder(nn.Module):
    """
    Standalone DINOv3 Visual Encoder:
    Maps input RGB batch [B, 3, 448, 784] -> L2 normalized CLS token [B, 384].
    """
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, image: torch.Tensor):
        feats = self.encoder.forward_features(image)
        cls_tok = feats[:, 0, :]
        return l2_norm(cls_tok)


# ==============================================================================
# Model Loading & Checkpoint Resolution
# ==============================================================================

def load_models(checkpoint_path: Path, dinov3_model_name: str = "vit_small_patch16_dinov3"):
    logger.info(f"Loading frozen DINOv3 backbone: '{dinov3_model_name}'...")
    dinov3 = timm.create_model(dinov3_model_name, pretrained=True)
    dinov3.eval()
    for p in dinov3.parameters():
        p.requires_grad = False

    state_dim = 4 * 384
    goal_dim = 384
    action_dim = 3
    actor = Actor(state_dim=state_dim, goal_dim=goal_dim, action_dim=action_dim)

    # Smart path resolution for leading slashes and common relative locations
    resolved_path = None
    if checkpoint_path is not None:
        raw_p = Path(checkpoint_path)
        if raw_p.exists():
            resolved_path = raw_p
        else:
            clean_str = str(checkpoint_path).lstrip("/\\")
            cand_paths = [
                Path.cwd() / clean_str,
                Path.home() / clean_str,
                Path(clean_str),
                Path.cwd() / "runs" / clean_str,
                Path.cwd() / "realworld" / "models" / clean_str,
            ]
            for cand in cand_paths:
                if cand.exists():
                    resolved_path = cand
                    logger.info(f"Resolved checkpoint path to: {resolved_path}")
                    break

    if resolved_path is not None and resolved_path.exists():
        logger.info(f"Loading trained Actor weights from: {resolved_path}")
        ckpt = torch.load(resolved_path, map_location="cpu")
        if "actor" in ckpt:
            actor.load_state_dict(ckpt["actor"])
        else:
            actor.load_state_dict(ckpt)
        logger.info("Successfully loaded Actor weights.")
    else:
        logger.warning(
            f"Checkpoint '{checkpoint_path}' not found! "
            "Using randomly initialized Actor. (Make sure you pass the correct path to your trained .pt checkpoint!)"
        )

    actor.eval()
    for p in actor.parameters():
        p.requires_grad = False

    return dinov3, actor


# ==============================================================================
# ONNX & OpenVINO Exporters
# ==============================================================================

def export_onnx(model: nn.Module, dummy_inputs: tuple, onnx_path: Path, input_names: list, output_names: list, dynamic_axes: dict):
    logger.info(f"Exporting ONNX model to: {onnx_path}")
    t0 = time.time()
    torch.onnx.export(
        model,
        dummy_inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )
    elapsed = time.time() - t0
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    data_file = onnx_path.with_name(onnx_path.name + ".data")
    if data_file.exists():
        size_mb += data_file.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX export completed in {elapsed:.2f}s | Size: {size_mb:.2f} MB")


def export_openvino(onnx_path: Path, ov_xml_path: Path, compress_fp16: bool = False):
    import openvino as ov
    logger.info(f"Compiling OpenVINO IR (compress_to_fp16={compress_fp16}) -> {ov_xml_path}")
    t0 = time.time()
    core = ov.Core()
    ov_model = core.read_model(str(onnx_path))
    ov.save_model(ov_model, str(ov_xml_path), compress_to_fp16=compress_fp16)
    elapsed = time.time() - t0
    bin_file = ov_xml_path.with_suffix(".bin")
    size_mb = (ov_xml_path.stat().st_size + (bin_file.stat().st_size if bin_file.exists() else 0)) / (1024 * 1024)
    logger.info(f"OpenVINO IR compiled in {elapsed:.2f}s | Size: {size_mb:.2f} MB")
    return core, ov_model


# ==============================================================================
# Verification & Benchmark
# ==============================================================================

def verify_and_benchmark(pyt_model: nn.Module, onnx_path: Path, ov_xml_path: Path, dummy_inputs_dict: dict, dummy_inputs_tuple: tuple):
    import onnxruntime as ort
    import openvino as ov

    logger.info("Executing Numerical Parity Verification...")

    # 1. PyTorch Eager reference
    with torch.no_grad():
        pyt_outs = pyt_model(*dummy_inputs_tuple)
    if not isinstance(pyt_outs, tuple):
        pyt_outs = (pyt_outs,)
    pyt_outs_np = [out.cpu().numpy() for out in pyt_outs]

    # 2. ONNX Runtime
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), sess_opts, providers=["CPUExecutionProvider"])
    ort_inputs = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in dummy_inputs_dict.items()}
    ort_outs = session.run(None, ort_inputs)

    # 3. OpenVINO
    core = ov.Core()
    ov_model = core.read_model(str(ov_xml_path))
    compiled = core.compile_model(ov_model, "CPU", {
        "PERFORMANCE_HINT": "LATENCY",
        "INFERENCE_NUM_THREADS": 4
    })
    req = compiled.create_infer_request()
    req.infer(ort_inputs)
    ov_outs = [req.get_output_tensor(i).data for i in range(len(pyt_outs_np))]

    # Parity checks
    for idx, (pyt_arr, ort_arr, ov_arr) in enumerate(zip(pyt_outs_np, ort_outs, ov_outs)):
        diff_ort = np.max(np.abs(pyt_arr - ort_arr))
        diff_ov = np.max(np.abs(pyt_arr - ov_arr))
        logger.info(f"  Output [{idx}]: ORT Max Diff = {diff_ort:.6e} | OV Max Diff = {diff_ov:.6e}")
        assert diff_ort < 1e-4, f"ONNX Runtime deviation exceeds threshold: {diff_ort}"
        assert diff_ov < 1e-4, f"OpenVINO deviation exceeds threshold: {diff_ov}"

    logger.info("[SUCCESS] Numerical Parity Verified: PyTorch == ONNX Runtime == OpenVINO (< 1e-4).")

    # Benchmarking
    logger.info("Running Performance Profiling (5 iterations)...")
    times_pyt, times_ort, times_ov = [], [], []

    # Warmup
    with torch.no_grad():
        _ = pyt_model(*dummy_inputs_tuple)
    _ = session.run(None, ort_inputs)
    req.infer(ort_inputs)

    for _ in range(5):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = pyt_model(*dummy_inputs_tuple)
        times_pyt.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        _ = session.run(None, ort_inputs)
        times_ort.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        req.infer(ort_inputs)
        times_ov.append((time.perf_counter() - t0) * 1000.0)

    m_pyt = np.mean(times_pyt)
    m_ort = np.mean(times_ort)
    m_ov = np.mean(times_ov)

    logger.info("-" * 65)
    logger.info(f"{'Engine':<20} | {'Mean Latency':<14} | {'FPS':<10} | {'Speedup':<8}")
    logger.info("-" * 65)
    logger.info(f"{'PyTorch Eager':<20} | {m_pyt:>10.1f} ms | {1000.0/m_pyt:>8.2f} | {'1.00x':>8}")
    logger.info(f"{'ONNX Runtime':<20} | {m_ort:>10.1f} ms | {1000.0/m_ort:>8.2f} | {m_pyt/m_ort:>7.2f}x")
    logger.info(f"{'Intel OpenVINO':<20} | {m_ov:>10.1f} ms | {1000.0/m_ov:>8.2f} | {m_pyt/m_ov:>7.2f}x")
    logger.info("-" * 65)


# ==============================================================================
# CLI Entrypoint
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Export fused DINOv3 + TD3-BC Actor pipeline to ONNX and OpenVINO")
    parser.add_argument("--checkpoint", type=str, default="models/run3/training/best_checkpoint.pt",
                        help="Path to trained Actor checkpoint (.pt)")
    parser.add_argument("--mode", type=str, choices=["step", "e2e", "encoder", "all"], default="step",
                        help="Model architecture to export ('step', 'e2e', 'encoder', or 'all')")
    parser.add_argument("--output-dir", type=str, default="realworld/models/fused",
                        help="Directory to store exported ONNX and OpenVINO files")
    parser.add_argument("--compress-fp16", action="store_true", default=False,
                        help="Compress OpenVINO IR weights to FP16")
    parser.add_argument("--dinov3-model", type=str, default="vit_small_patch16_dinov3",
                        help="timm model identifier for DINOv3")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    dinov3, actor = load_models(ckpt_path if ckpt_path.exists() else None, args.dinov3_model)

    modes_to_export = ["step", "e2e", "encoder"] if args.mode == "all" else [args.mode]

    for mode in modes_to_export:
        logger.info("=" * 70)
        logger.info(f"PROCESSING EXPORT MODE: '{mode.upper()}'")
        logger.info("=" * 70)

        if mode == "step":
            model = FusedStepPolicy(dinov3, actor)
            model.eval()

            dummy_img = torch.randn(1, 3, 448, 784, dtype=torch.float32)
            dummy_past = torch.randn(1, 1152, dtype=torch.float32)
            dummy_goal = F.normalize(torch.randn(1, 384, dtype=torch.float32), p=2, dim=-1)

            dummy_tuple = (dummy_img, dummy_past, dummy_goal)
            dummy_dict = {"current_image": dummy_img, "past_phi_stack": dummy_past, "goal_phi": dummy_goal}
            input_names = ["current_image", "past_phi_stack", "goal_phi"]
            output_names = ["action", "current_phi"]
            dynamic_axes = {
                "current_image": {0: "batch_size"},
                "past_phi_stack": {0: "batch_size"},
                "goal_phi": {0: "batch_size"},
                "action": {0: "batch_size"},
                "current_phi": {0: "batch_size"}
            }
            onnx_path = out_dir / "minav_fused_step_policy.onnx"
            ov_xml_path = out_dir / "minav_fused_step_policy.xml"

        elif mode == "e2e":
            model = FusedEndToEndPolicy(dinov3, actor)
            model.eval()

            dummy_stacked = torch.randn(1, 4, 3, 448, 784, dtype=torch.float32)
            dummy_goal_img = torch.randn(1, 3, 448, 784, dtype=torch.float32)

            dummy_tuple = (dummy_stacked, dummy_goal_img)
            dummy_dict = {"stacked_images": dummy_stacked, "goal_image": dummy_goal_img}
            input_names = ["stacked_images", "goal_image"]
            output_names = ["action", "similarity"]
            dynamic_axes = {
                "stacked_images": {0: "batch_size"},
                "goal_image": {0: "batch_size"},
                "action": {0: "batch_size"},
                "similarity": {0: "batch_size"}
            }
            onnx_path = out_dir / "minav_fused_e2e_policy.onnx"
            ov_xml_path = out_dir / "minav_fused_e2e_policy.xml"

        elif mode == "encoder":
            model = FusedDINOv3Encoder(dinov3)
            model.eval()

            dummy_img = torch.randn(1, 3, 448, 784, dtype=torch.float32)
            dummy_tuple = (dummy_img,)
            dummy_dict = {"image": dummy_img}
            input_names = ["image"]
            output_names = ["phi"]
            dynamic_axes = {
                "image": {0: "batch_size"},
                "phi": {0: "batch_size"}
            }
            onnx_path = out_dir / "minav_dinov3_encoder.onnx"
            ov_xml_path = out_dir / "minav_dinov3_encoder.xml"

        # 1. Export ONNX
        export_onnx(model, dummy_tuple, onnx_path, input_names, output_names, dynamic_axes)

        # 2. Export OpenVINO IR
        export_openvino(onnx_path, ov_xml_path, compress_fp16=args.compress_fp16)

        # 3. Verify & Benchmark
        verify_and_benchmark(model, onnx_path, ov_xml_path, dummy_dict, dummy_tuple)

    logger.info("=" * 70)
    logger.info("ALL EXPORTS AND VALIDATION CHECKS COMPLETED SUCCESSFULLY!")
    logger.info(f"Models directory: {out_dir}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
