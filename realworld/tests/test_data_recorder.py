"""
realworld/tests/test_data_recorder.py — Data recorder format tests.
====================================================================
Validates that DataRecorder produces output in the exact same format
as the MuJoCo pipeline, ensuring downstream compatibility.
"""

import sys
import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from shared.drive_command import DriveCommand
from realworld.src.data_recorder import DataRecorder


class TestDataRecorder:

    @pytest.fixture
    def recorder(self, tmp_dir):
        """Create a DataRecorder with small segment size for testing."""
        return DataRecorder(session_dir=tmp_dir / "exploration", segment_size=5)

    @pytest.fixture
    def sample_frame(self):
        """Small 48×64 RGB test image."""
        rng = np.random.default_rng(42)
        return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)

    def test_segment_directory_structure(self, recorder, sample_frame):
        """Creates segment_000/rgb/ directory and observations.csv."""
        recorder.open()
        cmd = DriveCommand(v_linear=1.0, v_lateral=0.0, v_angular=0.5)
        recorder.record_step(sample_frame, cmd, 1.0, 2.0, 0.5, 1000.0)
        recorder.close()

        seg_dir = recorder.session_dir / "segment_000"
        assert seg_dir.exists(), "segment_000 directory not created"
        assert (seg_dir / "rgb").exists(), "rgb subdirectory not created"
        assert (seg_dir / "observations.csv").exists(), "observations.csv not created"

    def test_csv_schema(self, recorder, sample_frame):
        """CSV has all required columns matching MuJoCo format."""
        recorder.open()
        cmd = DriveCommand(v_linear=1.0, v_lateral=0.2, v_angular=-0.3)
        recorder.record_step(sample_frame, cmd, 3.14, -2.71, 1.57, 1000.0)
        recorder.close()

        csv_path = recorder.session_dir / "segment_000" / "observations.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader)

        expected_headers = [
            "timestamp", "image_path",
            "target_vx", "target_vy", "target_wz",
            "cmd_vx", "cmd_vy", "cmd_wz",
            "executed_vx", "executed_vy", "executed_wz",
            "safety_blocked", "obstacle_distance"
        ]
        assert headers == expected_headers, f"CSV headers mismatch: {headers}"

    def test_csv_data_values(self, recorder, sample_frame):
        """CSV row contains correct data values."""
        recorder.open()
        cmd = DriveCommand(v_linear=1.234, v_lateral=-0.567, v_angular=0.890)
        recorder.record_step(sample_frame, cmd, 3.14, -2.71, 1.57, 1000.0, executed_cmd=cmd, safety_blocked=False)
        recorder.close()

        csv_path = recorder.session_dir / "segment_000" / "observations.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row = next(reader)

        # timestamp depends on real wall-clock time relative to recorder.open()
        # so we just assert it exists and is parseable
        assert type(float(row[0])) is float
        assert row[1] == "rgb/000000.jpg"
        assert row[2] == "1.234" # target_vx
        assert row[3] == "-0.567"
        assert row[4] == "0.890"
        assert row[5] == "1.234" # cmd_vx
        assert row[6] == "-0.567"
        assert row[7] == "0.890"
        assert row[8] == "1.234" # executed_vx
        assert row[9] == "-0.567"
        assert row[10] == "0.890"
        assert row[11] == "0", "safety_blocked should be 0"
        assert row[12] == "", "obstacle_distance should be empty string (nan)"

    def test_image_saved_as_jpg(self, recorder, sample_frame):
        """Saved files are valid JPEG images."""
        recorder.open()
        cmd = DriveCommand()
        recorder.record_step(sample_frame, cmd, 0.0, 0.0, 0.0, 1000.0)
        recorder.close()

        img_path = recorder.session_dir / "segment_000" / "rgb" / "000000.jpg"
        assert img_path.exists(), "Image file not saved"

        # Verify it's a valid image
        img = Image.open(img_path)
        assert img.size == (64, 48), f"Image size mismatch: {img.size}"

    def test_segment_rollover(self, recorder, sample_frame):
        """After segment_size steps, rolls to next segment directory."""
        recorder.open()
        cmd = DriveCommand()

        # Record 8 steps (segment_size=5, so we should get segment_000 + segment_001)
        for i in range(8):
            recorder.record_step(sample_frame, cmd, 0.0, 0.0, 0.0, 1000.0 + i)

        recorder.close()

        assert (recorder.session_dir / "segment_000").exists()
        assert (recorder.session_dir / "segment_001").exists()

        # segment_000 should have exactly 5 images
        seg0_images = list((recorder.session_dir / "segment_000" / "rgb").glob("*.jpg"))
        assert len(seg0_images) == 5, f"Expected 5 images in seg0, got {len(seg0_images)}"

        # segment_001 should have 3 images
        seg1_images = list((recorder.session_dir / "segment_001" / "rgb").glob("*.jpg"))
        assert len(seg1_images) == 3, f"Expected 3 images in seg1, got {len(seg1_images)}"

    def test_metadata_json_written(self, recorder, sample_frame):
        """exploration_metadata.json matches expected schema."""
        recorder.open()
        cmd = DriveCommand()
        for i in range(3):
            recorder.record_step(sample_frame, cmd, 0.0, 0.0, 0.0, 1000.0 + i)
        recorder.close(
            duration_minutes=2.0,
            control_freq_hz=10.0,
            total_steps_planned=100,
        )

        meta_path = recorder.session_dir / "exploration_metadata.json"
        assert meta_path.exists()

        with open(meta_path) as f:
            meta = json.load(f)

        assert meta["total_steps_recorded"] == 3
        assert meta["duration_minutes"] == 2.0
        assert meta["control_freq_hz"] == 10.0
        assert meta["robot_resets"] == 0
        assert meta["source"] == "realworld"
        assert isinstance(meta["segments"], list)


