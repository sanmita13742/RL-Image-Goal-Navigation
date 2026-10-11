"""
realworld/src/data_recorder.py — Segment-based exploration data recorder.
=========================================================================
Records exploration data in the EXACT same format as the MuJoCo pipeline,
so the downstream processing (DINOv3 → Hindsight → TD3+BC) works unchanged.

Output structure (identical to MuJoCo run_exploration.py):
    session_dir/
        segment_000/
            rgb/000000.png, 000001.png, ...
            observations.csv
        segment_001/
            ...
        exploration_metadata.json

CSV columns (strictly limited to real-world dataset requirement):
    timestamp, image_path,
    target_vx, target_vy, target_wz,
    cmd_vx, cmd_vy, cmd_wz,
    executed_vx, executed_vy, executed_wz,
    safety_blocked, obstacle_distance
"""

from __future__ import annotations

import csv
import json
import logging
import time
import queue
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from shared.drive_command import DriveCommand

logger = logging.getLogger(__name__)


class DataRecorder:
    """Records exploration data in MuJoCo-compatible format.

    Parameters
    ----------
    session_dir : Path
        Root directory for this exploration session.
    segment_size : int
        Number of steps per segment before rolling to the next.
    """

    def __init__(self, session_dir: Path, segment_size: int = 1000):
        self.session_dir = Path(session_dir)
        self.segment_size = segment_size
        
        self._queue = queue.Queue(maxsize=300)
        self._worker_thread = None
        self._stop_event = threading.Event()

        self._seg_id = 0
        self._seg_step = 0
        self._global_step = 0
        self._seg_start_global = 0
        self._segment_meta: list[dict] = []

        # Current segment file handles
        self._seg_dir: Optional[Path] = None
        self._rgb_dir: Optional[Path] = None
        self._csv_file = None
        self._csv_writer = None

        self._start_time: Optional[float] = None
        self._is_open = False

    def open(self) -> None:
        """Initialize the recorder and open the first segment."""
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing data (resume logic)
        metadata_path = self.session_dir / "exploration_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r") as f:
                    old_meta = json.load(f)
                self._segment_meta = old_meta.get("segments", [])
                if self._segment_meta:
                    last_seg = self._segment_meta[-1]
                    self._seg_id = int(last_seg["segment_id"].split("_")[1]) + 1
                    self._seg_start_global = last_seg["global_end"] + 1
                    self._global_step = self._seg_start_global
                    logger.info(
                        f"Resuming from segment_{self._seg_id:03d} "
                        f"at global step {self._seg_start_global}"
                    )
            except Exception as e:
                logger.warning(f"Could not parse existing metadata: {e}. Starting fresh.")
                self._segment_meta = []

        self._open_segment()
        self._start_time = time.time()
        self._is_open = True
        
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info(f"DataRecorder opened: {self.session_dir}")

    def _open_segment(self) -> None:
        """Open a new segment directory and CSV file."""
        seg_name = f"segment_{self._seg_id:03d}"
        self._seg_dir = self.session_dir / seg_name
        self._rgb_dir = self._seg_dir / "rgb"
        self._rgb_dir.mkdir(parents=True, exist_ok=True)

        csv_path = self._seg_dir / "observations.csv"
        self._csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp", "image_path",
            "pos_x", "pos_y", "pos_z", "yaw",
            "target_vx", "target_vy", "target_wz",
            "cmd_vx", "cmd_vy", "cmd_wz",
            "executed_vx", "executed_vy", "executed_wz",
            "safety_blocked", "obstacle_distance"
        ])
        self._seg_step = 0

    def _close_segment(self) -> None:
        """Close the current segment CSV file."""
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def record_step(
        self,
        rgb_frame: np.ndarray,
        cmd: DriveCommand,
        pos_x: float,
        pos_y: float,
        yaw: float,
        wall_time: float,
        pos_z: float = 0.0,
        executed_cmd: DriveCommand = None,
        safety_blocked: bool = False,
        target_cmd: DriveCommand = None,
        obstacle_distance: float = float('nan'),
        relevant_point_count: int = 0,
        cloud_age: float = float('nan'),
    ) -> int:
        """Record one exploration step by pushing to a background queue.

        Parameters
        ----------
        rgb_frame : np.ndarray
            RGB image for this step.
        cmd : DriveCommand
            Requested (post-LERP, pre-safety) command.
        pos_x, pos_y, pos_z : float
            Robot XYZ position from odometry. pos_z is height (metres above
            the odom origin — useful for ramp / multi-floor detection).
        yaw : float
            Robot heading in radians (from odom quaternion).
        wall_time : float
            Wall-clock timestamp of the camera frame.
        executed_cmd : DriveCommand, optional
            The command actually sent to the robot (post-safety gate).
        safety_blocked : bool
            Whether the safety gate blocked translation this step.
        target_cmd : DriveCommand, optional
            Raw pink-noise target (pre-LERP smoothing). Diagnostic only.
        obstacle_distance : float
            Distance to nearest obstacle in the safety corridor (m).
        relevant_point_count : int
            Number of LiDAR points in the safety corridor.
        cloud_age : float
            Age of the LiDAR point cloud (seconds).
        """
        if not self._is_open:
            raise RuntimeError("DataRecorder is not open. Call open() first.")
            
        try:
            self._queue.put_nowait((
                rgb_frame, cmd, pos_x, pos_y, pos_z, yaw, wall_time,
                executed_cmd, safety_blocked,
                target_cmd, obstacle_distance, relevant_point_count, cloud_age,
            ))
        except queue.Full:
            logger.error("DataRecorder queue is full! Dropping frame!")
            
        # The caller does not rely on the exact return value
        return 0

    def _worker_loop(self) -> None:
        """Background thread for handling disk I/O."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Unpack — handle legacy (8-tuple), old extended (12-tuple),
            # and current full format (13-tuple with pos_z)
            if len(item) == 13:
                (rgb_frame, cmd, pos_x, pos_y, pos_z, yaw, wall_time,
                 executed_cmd, safety_blocked,
                 target_cmd, obstacle_distance, relevant_point_count, cloud_age) = item
            elif len(item) == 12:
                (rgb_frame, cmd, pos_x, pos_y, yaw, wall_time,
                 executed_cmd, safety_blocked,
                 target_cmd, obstacle_distance, relevant_point_count, cloud_age) = item
                pos_z = 0.0
            else:
                (rgb_frame, cmd, pos_x, pos_y, yaw, wall_time, executed_cmd, safety_blocked) = item
                pos_z = 0.0
                target_cmd = None
                obstacle_distance = float('nan')
                relevant_point_count = 0
                cloud_age = float('nan')
            
            # Roll to new segment if needed
            if self._seg_step >= self.segment_size:
                self._close_segment()
                self._segment_meta.append({
                    "segment_id": f"segment_{self._seg_id:03d}",
                    "global_start": self._seg_start_global,
                    "global_end": self._global_step - 1,
                    "num_steps": self.segment_size,
                })
                logger.info(
                    f"[SEG DONE] segment_{self._seg_id:03d} | "
                    f"global {self._seg_start_global}-{self._global_step - 1}"
                )
                self._save_metadata()

                self._seg_id += 1
                self._seg_start_global = self._global_step
                self._open_segment()

            # Save RGB image
            img_filename = f"{self._seg_step:06d}.jpg"
            Image.fromarray(rgb_frame).save(self._rgb_dir / img_filename, format="JPEG", quality=90)

            # Compute elapsed time (analogous to sim_time)
            elapsed = wall_time - self._start_time if self._start_time else 0.0

            # Default executed_cmd to commanded if not provided
            ex = executed_cmd if executed_cmd is not None else cmd

            # Default target_cmd to commanded if not provided (legacy callers)
            tgt = target_cmd if target_cmd is not None else cmd

            # Write CSV row
            self._csv_writer.writerow([
                f"{elapsed:.3f}",
                f"rgb/{img_filename}",
                f"{pos_x:.4f}",
                f"{pos_y:.4f}",
                f"{pos_z:.4f}",
                f"{yaw:.6f}",
                f"{tgt.v_linear:.3f}",
                f"{tgt.v_lateral:.3f}",
                f"{tgt.v_angular:.3f}",
                f"{cmd.v_linear:.3f}",
                f"{cmd.v_lateral:.3f}",
                f"{cmd.v_angular:.3f}",
                f"{ex.v_linear:.3f}",
                f"{ex.v_lateral:.3f}",
                f"{ex.v_angular:.3f}",
                1 if safety_blocked else 0,
                f"{obstacle_distance:.3f}" if not (obstacle_distance != obstacle_distance) else "",
            ])

            self._global_step += 1
            self._seg_step += 1
            self._queue.task_done()

    def _save_metadata(self) -> None:
        """Save exploration_metadata.json (incremental)."""
        metadata_path = self.session_dir / "exploration_metadata.json"
        meta = {
            "run_id": self.session_dir.parent.name,
            "duration_minutes": None,  # Filled on close
            "control_freq_hz": None,
            "total_steps_planned": None,
            "total_steps_recorded": self._global_step,
            "segment_size": self.segment_size,
            "num_segments": len(self._segment_meta),
            "robot_resets": 0,
            "smoke_mode": False,
            "source": "realworld",
            "segments": self._segment_meta,
        }
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=4)

    def close(
        self,
        duration_minutes: float = None,
        control_freq_hz: float = None,
        total_steps_planned: int = None,
        is_smoke: bool = False,
    ) -> None:
        """Finalize the recording session."""
        if not self._is_open:
            return

        # Wait for all queued frames to be written
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._queue.join()

        # Stop worker thread
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join()

        # Close final segment
        self._close_segment()
        if self._seg_step > 0:
            self._segment_meta.append({
                "segment_id": f"segment_{self._seg_id:03d}",
                "global_start": self._seg_start_global,
                "global_end": self._global_step - 1,
                "num_steps": self._seg_step,
            })

        # Update metadata with final info
        metadata_path = self.session_dir / "exploration_metadata.json"
        meta = {
            "run_id": self.session_dir.parent.name,
            "duration_minutes": duration_minutes,
            "control_freq_hz": control_freq_hz,
            "total_steps_planned": total_steps_planned,
            "total_steps_recorded": self._global_step,
            "segment_size": self.segment_size,
            "num_segments": len(self._segment_meta),
            "robot_resets": 0,
            "smoke_mode": is_smoke,
            "source": "realworld",
            "segments": self._segment_meta,
        }
        with open(metadata_path, "w") as f:
            json.dump(meta, f, indent=4)

        self._is_open = False
        logger.info(
            f"DataRecorder closed. Total steps: {self._global_step}, "
            f"Segments: {len(self._segment_meta)}"
        )

    @property
    def global_step(self) -> int:
        """Current global step count."""
        return self._global_step
