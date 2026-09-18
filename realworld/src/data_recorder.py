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

CSV columns (matching MuJoCo):
    trajectory_id, global_step, segment_step, sim_time,
    linear_vel_cmd, lateral_vel_cmd, angular_vel_cmd,
    pos_x, pos_y, yaw, rgb_path, depth_path
"""

from __future__ import annotations

import csv
import json
import logging
import time
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
            "trajectory_id", "global_step", "segment_step", "sim_time",
            "linear_vel_cmd", "lateral_vel_cmd", "angular_vel_cmd",
            "executed_linear_vel", "executed_lateral_vel", "executed_angular_vel",
            "pos_x", "pos_y", "yaw", "rgb_path", "depth_path",
            "safety_intervention",
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
        executed_cmd: DriveCommand = None,
        safety_blocked: bool = False,
    ) -> int:
        """Record one exploration step.

        Parameters
        ----------
        rgb_frame : np.ndarray
            RGB image (H, W, 3) uint8.
        cmd : DriveCommand
            The commanded velocity (pre-safety-gate).
        pos_x, pos_y : float
            Robot position from odometry.
        yaw : float
            Robot heading from odometry.
        wall_time : float
            Wall-clock timestamp (seconds since epoch).
        executed_cmd : DriveCommand, optional
            The post-safety-gate velocity actually sent to the robot.
            If None, defaults to ``cmd`` (no intervention).
        safety_blocked : bool
            True if the safety gate blocked this step.

        Returns
        -------
        int
            The global_step index of this recorded step.
        """
        if not self._is_open:
            raise RuntimeError("DataRecorder is not open. Call open() first.")

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
        img_filename = f"{self._seg_step:06d}.png"
        Image.fromarray(rgb_frame).save(self._rgb_dir / img_filename)

        # Compute elapsed time (analogous to sim_time)
        elapsed = wall_time - self._start_time if self._start_time else 0.0

        # Default executed_cmd to commanded if not provided
        ex = executed_cmd if executed_cmd is not None else cmd

        # Write CSV row
        self._csv_writer.writerow([
            0,  # trajectory_id (single continuous trajectory)
            self._global_step,
            self._seg_step,
            f"{elapsed:.3f}",
            f"{cmd.v_linear:.3f}",
            f"{cmd.v_lateral:.3f}",
            f"{cmd.v_angular:.3f}",
            f"{ex.v_linear:.3f}",
            f"{ex.v_lateral:.3f}",
            f"{ex.v_angular:.3f}",
            f"{pos_x:.4f}",
            f"{pos_y:.4f}",
            f"{yaw:.4f}",
            f"rgb/{img_filename}",
            "",  # depth_path — not saved for real-world (LiDAR used directly)
            1 if safety_blocked else 0,
        ])

        step = self._global_step
        self._global_step += 1
        self._seg_step += 1

        return step

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
