import sys
import argparse
import yaml
import json
import time
import csv
import math
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import mujoco
from PIL import Image

from robot import RangerMiniV3Robot
from exploration_policies import PrimitiveExplorationPolicy
from test_env import build_world_xml
from scripts.pipeline_utils import setup_pipeline_logger, update_pipeline_state

def euler_from_quaternion(w, x, y, z):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1) if 'math' in sys.modules else np.arctan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2) if 'math' in sys.modules else np.arcsin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4) if 'math' in sys.modules else np.arctan2(t3, t4)
    return roll, pitch, yaw

def save_depth(arr: np.ndarray, path: str) -> None:
    d_min, d_max = arr.min(), arr.max()
    norm = ((arr - d_min) / (d_max - d_min) * 255).astype(np.uint8) \
           if d_max > d_min else np.zeros_like(arr, dtype=np.uint8)
    Image.fromarray(norm, mode="L").save(path)

def open_segment(session_dir: Path, seg_id: int):
    seg_name  = f"segment_{seg_id:03d}"
    seg_dir   = session_dir / seg_name
    rgb_dir   = seg_dir / "rgb"
    depth_dir = seg_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    csv_file = open(seg_dir / "observations.csv", mode="w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "trajectory_id", "global_step", "segment_step", "sim_time",
        "linear_vel_cmd", "lateral_vel_cmd", "angular_vel_cmd",
        "pos_x", "pos_y", "yaw", "rgb_path", "depth_path",
    ])
    return seg_dir, rgb_dir, depth_dir, csv_file, writer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--map", type=str, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    logger = setup_pipeline_logger(run_dir, "exploration")
    update_pipeline_state(run_dir, "exploration", "running")

    try:
        with open(ROOT / args.config) as f:
            config = yaml.safe_load(f)

        is_smoke = args.smoke or config.get("smoke", {}).get("enabled", False)
        
        if is_smoke:
            duration_minutes = config["smoke"]["exploration_minutes"]
            logger.warning("SMOKE MODE ACTIVE — limits enforced")
        else:
            duration_minutes = config["exploration"]["duration_minutes"]

        control_freq = config["exploration"]["control_freq_hz"]
        segment_size = config["exploration"]["segment_size"]
        out_subdir = config["exploration"]["output_subdir"]
        
        total_steps = int(duration_minutes * 60 * control_freq)
        
        session_dir = run_dir / out_subdir
        session_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting exploration. Duration: {duration_minutes}m, Steps: {total_steps}")
        
        # We assume map is XML. If user provided a path to an XML, we load it.
        # But MINav randomly generates it. We'll use the user provided XML if valid.
        map_path = Path(args.map)
        if not map_path.exists():
            # Fallback to random if user map not found
            world_xml_content = build_world_xml()
            world_xml = session_dir / "_temp_world.xml"
            world_xml.write_text(world_xml_content, encoding="utf-8")
            map_path = world_xml
            
        robot = RangerMiniV3Robot()
        robot.load(map_path)

        robot.data.qpos[0] = -4.5
        robot.data.qpos[1] = 0.0
        robot.data.qpos[3] = 0.7071068
        robot.data.qpos[4] = 0.0
        robot.data.qpos[5] = 0.0
        robot.data.qpos[6] = 0.7071068
        mujoco.mj_forward(robot.model, robot.data)

        renderer_rgb = mujoco.Renderer(robot.model, height=240, width=320)
        renderer_depth = mujoco.Renderer(robot.model, height=60, width=640)
        renderer_depth.enable_depth_rendering()

        sim_dt = robot.model.opt.timestep
        sim_steps_per_control = int(1.0 / (control_freq * sim_dt))

        policy = PrimitiveExplorationPolicy(control_freq, beta=1)
        
        seg_id = 0
        seg_dir, rgb_dir, depth_dir, csv_file, writer = open_segment(session_dir, seg_id)
        seg_step = 0
        seg_start_global = 0
        
        segment_meta = []
        
        start_time = time.time()
        
        for global_step in range(total_steps):
            if seg_step == segment_size:
                csv_file.close()
                segment_meta.append({
                    "segment_id": f"segment_{seg_id:03d}",
                    "global_start": seg_start_global,
                    "global_end": global_step - 1,
                    "num_steps": segment_size
                })
                logger.info(f"[SEG DONE] segment_{seg_id:03d} | global {seg_start_global}-{global_step-1}")
                
                seg_id += 1
                seg_step = 0
                seg_start_global = global_step
                seg_dir, rgb_dir, depth_dir, csv_file, writer = open_segment(session_dir, seg_id)
                
            renderer_rgb.update_scene(robot.data, camera="front_cam")
            rgb_img = renderer_rgb.render()
            
            renderer_depth.update_scene(robot.data, camera="lidar_cam")
            depth_img = renderer_depth.render()
            
            pos_x, pos_y, _ = robot.data.qpos[0:3]
            qw, qx, qy, qz = robot.data.qpos[3:7]
            _, _, yaw = euler_from_quaternion(qw, qx, qy, qz)
            
            cmd, prim = policy.get_action(depth_img, pos_x, pos_y, yaw)
            
            img_filename = f"{seg_step:06d}.png"
            Image.fromarray(rgb_img).save(rgb_dir / img_filename)
            save_depth(depth_img, str(depth_dir / img_filename))
            
            sim_time = robot.data.time
            writer.writerow([
                0, global_step, seg_step, f"{sim_time:.3f}",
                f"{cmd.v_linear:.3f}", f"{cmd.v_lateral:.3f}", f"{cmd.v_angular:.3f}",
                f"{pos_x:.4f}", f"{pos_y:.4f}", f"{yaw:.4f}",
                f"rgb/{img_filename}", f"depth/{img_filename}"
            ])
            
            if global_step % 100 == 0:
                logger.info(f"[g={global_step:06d}] {prim.name} v={cmd.v_linear:+.2f} w={cmd.v_angular:+.2f}")
                
            for _ in range(sim_steps_per_control):
                robot.apply_command(cmd)
                robot.step()
                
            seg_step += 1
            
        csv_file.close()
        segment_meta.append({
            "segment_id": f"segment_{seg_id:03d}",
            "global_start": seg_start_global,
            "global_end": total_steps - 1,
            "num_steps": seg_step
        })
        
        meta = {
            "run_id": run_dir.name,
            "duration_minutes": duration_minutes,
            "control_freq_hz": control_freq,
            "total_steps_planned": total_steps,
            "total_steps_recorded": total_steps,
            "segment_size": segment_size,
            "num_segments": len(segment_meta),
            "robot_resets": 0,
            "smoke_mode": is_smoke,
            "segments": segment_meta
        }
        with open(session_dir / "exploration_metadata.json", "w") as f:
            json.dump(meta, f, indent=4)
            
        logger.info("Exploration phase completed successfully.")
        update_pipeline_state(run_dir, "exploration", "completed")
        
    except Exception as e:
        logger.error(f"Exploration failed: {e}", exc_info=True)
        update_pipeline_state(run_dir, "exploration", "failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
