#!/usr/bin/env python3
"""
realworld/tests/test_deployment_metrics.py — Unit Tests for Deployment Metrics Logger.
====================================================================================
Tests SR, Time, and STL metric calculations and export formatting.
"""

import sys
import shutil
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from realworld.src.deployment_metrics_logger import DeploymentMetricsLogger

TEST_LOG_DIR = ROOT / "results" / "test_deployment_metrics_temp"


@pytest.fixture(autouse=True)
def clean_test_dir():
    if TEST_LOG_DIR.exists():
        shutil.rmtree(TEST_LOG_DIR)
    yield
    if TEST_LOG_DIR.exists():
        shutil.rmtree(TEST_LOG_DIR)


def test_metrics_logger_single_episode_success():
    logger = DeploymentMetricsLogger(log_dir=TEST_LOG_DIR)

    ep_id = logger.start_episode(
        split="simple",
        goal_image="goal_test.png",
        max_steps=100,
        control_freq=10.0,
        optimal_time_sec=5.0,
    )
    assert ep_id.startswith("ep_")

    # Log 3 steps
    logger.log_step(step=0, similarity=0.2, cmd_vx=0.5, cmd_vy=0.0, cmd_wz=0.0, odom_x=0.0, odom_y=0.0)
    logger.log_step(step=1, similarity=0.5, cmd_vx=0.5, cmd_vy=0.0, cmd_wz=0.0, odom_x=0.05, odom_y=0.0)
    logger.log_step(step=2, similarity=0.8, cmd_vx=0.2, cmd_vy=0.0, cmd_wz=0.0, odom_x=0.10, odom_y=0.0)

    # End with success
    record = logger.end_episode(success=True, termination_reason="goal_reached", final_similarity=0.82)

    assert record["success"] == 1
    assert record["stl_pct"] > 0.0
    assert record["path_length_meters"] > 0.09
    assert (TEST_LOG_DIR / "deployment_episodes.csv").exists()
    assert (TEST_LOG_DIR / "episodes" / f"{ep_id}_telemetry.csv").exists()


def test_metrics_logger_failure_stl_zero():
    logger = DeploymentMetricsLogger(log_dir=TEST_LOG_DIR)

    ep_id = logger.start_episode(split="complex", max_steps=100, control_freq=10.0, optimal_time_sec=10.0)
    logger.log_step(step=0, similarity=0.1, cmd_vx=0.5, cmd_vy=0.0, cmd_wz=0.0)
    record = logger.end_episode(success=False, termination_reason="timeout", final_similarity=0.35)

    assert record["success"] == 0
    assert record["stl_pct"] == 0.0
    assert record["stl_budget_pct"] == 0.0


def test_summary_report_aggregation():
    logger = DeploymentMetricsLogger(log_dir=TEST_LOG_DIR)

    # 1 success, 1 failure in 'standard' split
    logger.start_episode(split="standard", max_steps=50, control_freq=10.0, optimal_time_sec=4.0)
    logger.log_step(step=0, similarity=0.7, cmd_vx=0.5, cmd_vy=0.0, cmd_wz=0.0)
    logger.end_episode(success=True)

    logger.start_episode(split="standard", max_steps=50, control_freq=10.0, optimal_time_sec=4.0)
    logger.log_step(step=0, similarity=0.2, cmd_vx=0.5, cmd_vy=0.0, cmd_wz=0.0)
    logger.end_episode(success=False)

    summary = logger.generate_summary_report()

    assert "standard" in summary["splits"]
    std_data = summary["splits"]["standard"]
    assert std_data["n_episodes"] == 2
    assert std_data["sr_pct"] == 50.0

    # Ensure artifacts generated
    assert (TEST_LOG_DIR / "deployment_metrics_table.tex").exists()
    assert (TEST_LOG_DIR / "deployment_metrics_table.md").exists()
    assert (TEST_LOG_DIR / "deployment_metrics_summary.json").exists()
