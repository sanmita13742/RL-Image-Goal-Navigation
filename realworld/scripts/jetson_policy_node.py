#!/usr/bin/env python3
"""
realworld/scripts/jetson_policy_node.py — GPU AI Inference Server for Jetson.
=============================================================================
Runs on the NVIDIA Jetson on the internal LAN.
Subscribes to camera frames from the robot, runs GPU-accelerated DINOv3 + TD3+BC,
and publishes normalized action vectors back to the robot CPU.

Pipeline:
  1. Subscribes to /camera/color/image_raw (sensor_msgs/msg/Image)
  2. Runs Frozen DINOv3 visual encoder on CUDA (10-20ms per frame)
  3. Maintains 4-frame FIFO history [t-3, t-2, t-1, t0]
  4. Runs TD3+BC Actor network on CUDA (1-2ms)
  5. Computes cosine similarity to goal image
  6. Publishes /minav/policy_action (geometry_msgs/msg/TwistStamped)
     - linear.x: normalized vx in [-1, 1]
     - linear.y: normalized vy in [-1, 1]
     - angular.z: normalized omega in [-1, 1]
     - linear.z: cosine similarity (scalar telemetry)
     - header.stamp: capture timestamp for latency tracking
"""

import sys
import argparse
import logging
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
from PIL import Image as PILImage

# Project root resolution
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

MUJOCO_ROOT = ROOT / "simulation" / "ranger_mini_v3"
sys.path.insert(0, str(MUJOCO_ROOT))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image as RosImage
from geometry_msgs.msg import TwistStamped

from src.rl.networks import Actor
from src.data.dinov3_encoder import FrozenDINOv3

logger = logging.getLogger("jetson_policy_node")


def parse_args():
    parser = argparse.ArgumentParser(description="MINav Jetson GPU Inference Node")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to actor checkpoint (.pt)")
    parser.add_argument("--goal-image", type=str, required=True, help="Path to target goal RGB image")
    parser.add_argument("--image-topic", type=str, default="/camera/color/image_raw",
                        help="Camera image topic from robot CPU (default: /camera/color/image_raw)")
    parser.add_argument("--action-topic", type=str, default="/minav/policy_action",
                        help="Topic to publish normalized actions (default: /minav/policy_action)")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device ('cuda' or 'cpu')")
    return parser.parse_args()


class JetsonInferenceNode(Node):
    def __init__(self, encoder, actor, goal_phi, action_topic, image_topic, device):
        super().__init__("jetson_policy_node")
        self.encoder = encoder
        self.actor = actor
        self.device = device
        self.goal_phi = goal_phi  # [384] tensor
        self.goal_phi_batch = goal_phi.unsqueeze(0)  # [1, 384]

        self.phi_buffer = deque(maxlen=4)
        self.step_count = 0

        # Action Publisher
        self.action_pub = self.create_publisher(TwistStamped, action_topic, 10)

        # Image Subscriber (matching RealSense TRANSIENT_LOCAL/RELIABLE QoS)
        image_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=5,
        )
        self.image_sub = self.create_subscription(
            RosImage, image_topic, self.image_callback, image_qos
        )

        self.get_logger().info(f"Jetson AI Inference Node active.")
        self.get_logger().info(f"  Input Image: {image_topic}")
        self.get_logger().info(f"  Output Action: {action_topic}")
        self.get_logger().info(f"  Inference Device: {device}")

    def image_callback(self, msg: RosImage):
        t0 = time.time()
        # Decode raw image data -> RGB numpy array
        # RealSense color encoding is typically rgb8 or bgr8
        if msg.encoding == "rgb8":
            rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        elif msg.encoding == "bgr8":
            bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            rgb = bgr[:, :, ::-1].copy()
        else:
            self.get_logger().error(f"Unsupported image encoding: {msg.encoding}", throttle_duration_sec=5.0)
            return

        # 1. GPU Visual Feature Encoding (DINOv3)
        phi_tensor = self.encoder.encode_frame(rgb)  # [384] tensor on self.device
        self.phi_buffer.append(phi_tensor)

        if len(self.phi_buffer) < 4:
            self.get_logger().info(
                f"Buffering initial temporal states: {len(self.phi_buffer)}/4 frames...",
                throttle_duration_sec=1.0
            )
            return

        # 2. Construct 4-frame temporal state [1, 1536]
        state = torch.cat(list(self.phi_buffer), dim=0).unsqueeze(0)

        # 3. Policy forward pass -> normalized action in [-1, 1]^3
        with torch.inference_mode():
            action_norm = self.actor(state, self.goal_phi_batch)  # [1, 3]

        norm_acts = action_norm[0].cpu().numpy()
        vx_norm = float(norm_acts[0])
        vy_norm = float(norm_acts[1])
        omega_norm = float(norm_acts[2])

        # 4. Telemetry: Cosine similarity
        sim = float(torch.dot(phi_tensor, self.goal_phi).item())

        # 5. Publish TwistStamped with stamp from the original image for latency calculation
        out_msg = TwistStamped()
        out_msg.header.stamp = msg.header.stamp
        out_msg.header.frame_id = "base_link"
        out_msg.twist.linear.x = vx_norm
        out_msg.twist.linear.y = vy_norm
        out_msg.twist.angular.z = omega_norm
        out_msg.twist.linear.z = sim  # embed cosine similarity for gateway stop logic

        self.action_pub.publish(out_msg)
        dt_ms = (time.time() - t0) * 1000

        self.step_count += 1
        if self.step_count % 10 == 0:
            self.get_logger().info(
                f"Step {self.step_count:4d} | Latency: {dt_ms:4.1f}ms | sim: {sim:.3f} | "
                f"norm: [vx={vx_norm:+.2f}, vy={vy_norm:+.2f}, ω={omega_norm:+.2f}]"
            )


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

    # 1. Resolve compute device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable on this system. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"AI Compute device: {device}")

    # 2. Load DINOv3 visual encoder
    logger.info("Initializing frozen DINOv3 visual encoder (vit_small_patch16_dinov3)...")
    encoder = FrozenDINOv3(device=str(device))

    # 3. Encode goal image
    goal_path = Path(args.goal_image)
    if not goal_path.exists():
        logger.error(f"Goal image not found: {goal_path}")
        return
    logger.info(f"Encoding goal image: {goal_path}")
    goal_rgb = np.array(PILImage.open(goal_path).convert("RGB"))
    goal_phi = encoder.encode_frame(goal_rgb)
    logger.info(f"Goal embedding ready. Shape: {goal_phi.shape}, L2-norm: {torch.norm(goal_phi).item():.3f}")

    # 4. Load Actor network
    state_dim = 4 * 384
    goal_dim = 384
    action_dim = 3

    actor = Actor(state_dim, goal_dim, action_dim).to(device)
    logger.info(f"Loading actor weights from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    else:
        actor.load_state_dict(ckpt)
    actor.eval()
    logger.info("Actor network initialized in eval mode.")

    rclpy.init()
    node = JetsonInferenceNode(
        encoder=encoder,
        actor=actor,
        goal_phi=goal_phi,
        action_topic=args.action_topic,
        image_topic=args.image_topic,
        device=device,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.warning("Jetson inference server stopped by operator.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
