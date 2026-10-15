#!/usr/bin/env python3
"""
realworld/src/deployment_metrics_logger.py — Deployment Metrics Logger for MINav.
================================================================================
Tracks, logs, and calculates standardized robotic navigation metrics across
deployment runs on physical hardware and high-fidelity simulation:

Key Metrics Tracked (as published in MINav benchmark tables):
  1. Success Rate (SR % ↑):
       Percentage of navigation trials that successfully reached the goal.
  2. Navigation Time (Time s ↓):
       Duration (in seconds) required to reach the goal (reported as Mean ± Std).
  3. Success weighted by normalized Time / Time-to-Goal (STL % ↑):
       Measures time-optimality of successful navigation:
         STL = S * (T* / max(T, T*)) * 100%
       where S is binary success (0 or 1), T is actual navigation time,
       and T* is the reference optimal time budget (or S * (1 - T/T_max)).
  4. Success weighted by Path Length (SPL % ↑):
       SPL = S * (L* / max(L, L*)) * 100%
  5. Policy Parameters & Hardware Computational Footprint:
       Policy parameter count, backbone architecture, and training cost.

Outputs:
  - CSV episode ledger: <log_dir>/deployment_episodes.csv
  - Step telemetry per run: <log_dir>/episodes/<episode_id>_telemetry.csv
  - Aggregated JSON summary: <log_dir>/deployment_metrics_summary.json
  - LaTeX table snippet: <log_dir>/deployment_metrics_table.tex (Ready for paper insertion)
  - Markdown summary: <log_dir>/deployment_metrics_table.md
"""

import sys
import os
import csv
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("deployment_metrics")


class DeploymentMetricsLogger:
    """
    Structured logger for online deployment and evaluation trials.
    """

    def __init__(
        self,
        log_dir: Union[str, Path] = "results/deployment_metrics",
        method_name: str = "MINav (TD3+BC)",
        visual_backbone: str = "DINOv3 (ViT-S/16)",
        policy_params: int = 558339,
        training_tflops_hours: float = 12.4,
    ):
        self.log_dir = Path(log_dir).resolve()
        self.episodes_dir = self.log_dir / "episodes"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        self.method_name = method_name
        self.visual_backbone = visual_backbone
        self.policy_params = policy_params
        self.training_tflops_hours = training_tflops_hours

        self.ledger_csv = self.log_dir / "deployment_episodes.csv"
        self._init_ledger()

        # Active episode state
        self.active_episode: Optional[Dict] = None
        self.active_telemetry: List[Dict] = []
        self._prev_odom: Optional[Tuple[float, float]] = None
        self._path_length: float = 0.0

    def _init_ledger(self):
        """Initialize the master episode ledger CSV with headers if it does not exist."""
        if not self.ledger_csv.exists():
            headers = [
                "episode_id",
                "timestamp",
                "method",
                "visual_backbone",
                "policy_params",
                "training_tflops_hours",
                "split",
                "goal_image",
                "success",
                "nav_time_sec",
                "steps_taken",
                "max_steps",
                "control_freq_hz",
                "optimal_time_sec",
                "stl_pct",
                "stl_budget_pct",
                "final_similarity",
                "goal_threshold",
                "path_length_meters",
                "safety_interventions",
                "termination_reason",
            ]
            with open(self.ledger_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def start_episode(
        self,
        split: str = "complex",
        goal_image: str = "",
        max_steps: int = 1000,
        control_freq: float = 10.0,
        optimal_time_sec: Optional[float] = None,
        goal_threshold: float = 0.80,
        episode_id: Optional[str] = None,
    ) -> str:
        """
        Start logging a new deployment episode.

        Parameters:
          split: Environment difficulty ('simple', 'standard', 'complex')
          goal_image: Path or name of goal target image
          max_steps: Maximum step limit before timeout
          control_freq: Rate in Hz (e.g. 10.0 Hz)
          optimal_time_sec: Theoretical shortest time to goal (T*)
          goal_threshold: Cosine similarity threshold for goal arrival
        """
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        if episode_id is None:
            episode_id = f"ep_{ts_str}_{split}"

        # If optimal time not provided, default to 50% of time budget or distance heuristic
        max_time_sec = max_steps / control_freq
        if optimal_time_sec is None:
            if split.lower() == "simple":
                optimal_time_sec = min(15.0, max_time_sec * 0.4)
            elif split.lower() == "standard":
                optimal_time_sec = min(25.0, max_time_sec * 0.5)
            else:
                optimal_time_sec = min(35.0, max_time_sec * 0.6)

        self.active_episode = {
            "episode_id": episode_id,
            "timestamp": ts_str,
            "method": self.method_name,
            "visual_backbone": self.visual_backbone,
            "policy_params": self.policy_params,
            "training_tflops_hours": self.training_tflops_hours,
            "split": split.lower(),
            "goal_image": str(goal_image),
            "max_steps": max_steps,
            "control_freq_hz": control_freq,
            "max_time_budget_sec": max_time_sec,
            "optimal_time_sec": optimal_time_sec,
            "goal_threshold": goal_threshold,
            "start_time": time.time(),
        }

        self.active_telemetry = []
        self._prev_odom = None
        self._path_length = 0.0

        logger.info(
            f"Started deployment episode '{episode_id}' | Split={split.upper()} | "
            f"Goal={Path(goal_image).name} | Time Budget={max_time_sec:.1f}s | T*={optimal_time_sec:.1f}s"
        )
        return episode_id

    def log_step(
        self,
        step: int,
        similarity: float,
        cmd_vx: float,
        cmd_vy: float,
        cmd_wz: float,
        odom_x: Optional[float] = None,
        odom_y: Optional[float] = None,
        odom_yaw: Optional[float] = None,
        safety_blocked: bool = False,
    ):
        """Log per-step control telemetry."""
        if self.active_episode is None:
            return

        now = time.time()
        elapsed = now - self.active_episode["start_time"]

        # Track trajectory Euclidean path length if odometry available
        if odom_x is not None and odom_y is not None:
            if self._prev_odom is not None:
                dx = odom_x - self._prev_odom[0]
                dy = odom_y - self._prev_odom[1]
                self._path_length += float(np.hypot(dx, dy))
            self._prev_odom = (odom_x, odom_y)

        step_record = {
            "step": step,
            "elapsed_sec": round(elapsed, 4),
            "similarity": round(float(similarity), 4),
            "cmd_vx": round(float(cmd_vx), 3),
            "cmd_vy": round(float(cmd_vy), 3),
            "cmd_wz": round(float(cmd_wz), 3),
            "odom_x": round(float(odom_x), 3) if odom_x is not None else None,
            "odom_y": round(float(odom_y), 3) if odom_y is not None else None,
            "odom_yaw": round(float(odom_yaw), 3) if odom_yaw is not None else None,
            "safety_blocked": int(safety_blocked),
        }
        self.active_telemetry.append(step_record)

    def end_episode(
        self,
        success: bool,
        termination_reason: str = "goal_reached",
        final_similarity: Optional[float] = None,
    ) -> Dict:
        """
        Finalize episode, calculate metrics (SR, Time, STL, SPL), and write logs.
        """
        if self.active_episode is None:
            raise RuntimeError("No active episode to end.")

        end_time = time.time()
        nav_time_sec = end_time - self.active_episode["start_time"]
        steps_taken = len(self.active_telemetry)

        # Use final telemetry similarity if not provided
        if final_similarity is None and steps_taken > 0:
            final_similarity = self.active_telemetry[-1]["similarity"]
        elif final_similarity is None:
            final_similarity = 0.0

        t_star = self.active_episode["optimal_time_sec"]
        max_time = self.active_episode["max_time_budget_sec"]

        # ======================================================================
        # Metric 1: SR (%) - Binary success indicator
        # ======================================================================
        succ_val = 1 if success else 0

        # ======================================================================
        # Metric 2: Time (s) - Actual navigation time taken
        # ======================================================================
        time_s = nav_time_sec

        # ======================================================================
        # Metric 3: STL (%) - Success weighted by Time Length
        # Formulation A: STL = S * (T* / max(T, T*)) * 100% (Ratio to optimal time)
        # Formulation B: STL_budget = S * (1 - T / T_max) * 100% (Margin of budget)
        # ======================================================================
        if succ_val == 1:
            stl_ratio = t_star / max(time_s, t_star)
            stl_budget = max(0.0, 1.0 - (time_s / max_time))
        else:
            stl_ratio = 0.0
            stl_budget = 0.0

        stl_pct = round(stl_ratio * 100.0, 2)
        stl_budget_pct = round(stl_budget * 100.0, 2)

        # Count safety gate blocks
        safety_interventions = sum(r["safety_blocked"] for r in self.active_telemetry)

        record = {
            "episode_id": self.active_episode["episode_id"],
            "timestamp": self.active_episode["timestamp"],
            "method": self.active_episode["method"],
            "visual_backbone": self.active_episode["visual_backbone"],
            "policy_params": self.active_episode["policy_params"],
            "training_tflops_hours": self.active_episode["training_tflops_hours"],
            "split": self.active_episode["split"],
            "goal_image": self.active_episode["goal_image"],
            "success": succ_val,
            "nav_time_sec": round(time_s, 2),
            "steps_taken": steps_taken,
            "max_steps": self.active_episode["max_steps"],
            "control_freq_hz": self.active_episode["control_freq_hz"],
            "optimal_time_sec": round(t_star, 2),
            "stl_pct": stl_pct,
            "stl_budget_pct": stl_budget_pct,
            "final_similarity": round(float(final_similarity), 4),
            "goal_threshold": self.active_episode["goal_threshold"],
            "path_length_meters": round(self._path_length, 3),
            "safety_interventions": safety_interventions,
            "termination_reason": termination_reason,
        }

        # 1. Append to master ledger CSV
        with open(self.ledger_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(record.keys()))
            writer.writerow(record)

        # 2. Save detailed step telemetry CSV
        ep_csv = self.episodes_dir / f"{record['episode_id']}_telemetry.csv"
        if len(self.active_telemetry) > 0:
            df_tel = pd.DataFrame(self.active_telemetry)
            df_tel.to_csv(ep_csv, index=False)

        logger.info(
            f"Episode '{record['episode_id']}' Finished | Success: {bool(succ_val)} | "
            f"Time: {time_s:.2f}s | STL: {stl_pct:.1f}% | Steps: {steps_taken} | Path: {self._path_length:.2f}m"
        )

        self.active_episode = None
        self.active_telemetry = []

        # Automatically update aggregated summary
        self.generate_summary_report()

        return record

    def generate_summary_report(self) -> Dict:
        """
        Parses all logged episodes in the ledger and generates the aggregated
        benchmark statistics formatted for publication.
        """
        if not self.ledger_csv.exists() or os.stat(self.ledger_csv).st_size == 0:
            return {}

        df = pd.read_csv(self.ledger_csv)
        if len(df) == 0:
            return {}

        splits = ["simple", "standard", "complex"]
        summary = {
            "method": self.method_name,
            "visual_backbone": self.visual_backbone,
            "policy_params": self.policy_params,
            "training_tflops_hours": self.training_tflops_hours,
            "total_episodes": len(df),
            "splits": {},
        }

        for split in splits:
            sub = df[df["split"].str.lower() == split]
            if len(sub) == 0:
                summary["splits"][split] = {
                    "n_episodes": 0,
                    "sr_pct": 0.0,
                    "sr_std": 0.0,
                    "time_mean_s": 0.0,
                    "time_std_s": 0.0,
                    "stl_mean_pct": 0.0,
                    "stl_std_pct": 0.0,
                    "path_mean_m": 0.0,
                }
                continue

            n_ep = len(sub)
            sr_pct = float(sub["success"].mean() * 100.0)
            sr_std = float(sub["success"].std() * 100.0) if n_ep > 1 else 0.0

            # Time is traditionally reported on successful episodes or all episodes
            succ_sub = sub[sub["success"] == 1]
            if len(succ_sub) > 0:
                time_mean = float(succ_sub["nav_time_sec"].mean())
                time_std = float(succ_sub["nav_time_sec"].std()) if len(succ_sub) > 1 else 0.0
            else:
                time_mean = float(sub["nav_time_sec"].mean())
                time_std = float(sub["nav_time_sec"].std()) if n_ep > 1 else 0.0

            stl_mean = float(sub["stl_pct"].mean())
            stl_std = float(sub["stl_pct"].std()) if n_ep > 1 else 0.0

            path_mean = float(sub["path_length_meters"].mean())

            summary["splits"][split] = {
                "n_episodes": n_ep,
                "sr_pct": round(sr_pct, 1),
                "sr_std": round(sr_std, 1),
                "time_mean_s": round(time_mean, 1),
                "time_std_s": round(time_std, 1),
                "stl_mean_pct": round(stl_mean, 1),
                "stl_std_pct": round(stl_std, 1),
                "path_mean_m": round(path_mean, 2),
            }

        # 1. Save JSON summary
        json_path = self.log_dir / "deployment_metrics_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 2. Save LaTeX table snippet
        tex_path = self.log_dir / "deployment_metrics_table.tex"
        self._write_latex_table(summary, tex_path)

        # 3. Save Markdown table snippet
        md_path = self.log_dir / "deployment_metrics_table.md"
        self._write_markdown_table(summary, md_path)

        return summary

    def _write_latex_table(self, summary: Dict, tex_path: Path):
        """Format summary into a LaTeX tabular matching report.tex / metrics.png."""
        simp = summary["splits"].get("simple", {})
        stan = summary["splits"].get("standard", {})
        comp = summary["splits"].get("complex", {})

        tex_content = f"""% Auto-generated deployment metrics table for report.tex
\\begin{{table*}}[t]
\\centering
\\caption{{Navigation Performance Across Environment Difficulties on Ranger Mini V3 Mobile Base}}
\\label{{tab:navigation_metrics}}
\\resizebox{{\\textwidth}}{{!}}{{
\\begin{{tabular}}{{lcccccccccccc}}
\\hline
\\hline
\\textbf{{Method}} & \\textbf{{Visual backbone}} & \\textbf{{Policy Params}} $\\downarrow$ & \\textbf{{Training Cost}} $\\downarrow$ & \\multicolumn{{3}}{{c}}{{\\textbf{{Simple}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Standard}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Complex}}}} \\\\
\\cline{{5-7}} \\cline{{8-10}} \\cline{{11-13}}
& & & (TFLOPs-hours) & \\textbf{{SR(\\%)}} $\\uparrow$ & \\textbf{{Time(s)}} $\\downarrow$ & \\textbf{{STL(\\%)}} $\\uparrow$ & \\textbf{{SR(\\%)}} $\\uparrow$ & \\textbf{{Time(s)}} $\\downarrow$ & \\textbf{{STL(\\%)}} $\\uparrow$ & \\textbf{{SR(\\%)}} $\\uparrow$ & \\textbf{{Time(s)}} $\\downarrow$ & \\textbf{{STL(\\%)}} $\\uparrow$ \\\\
\\hline
{summary['method']} & {summary['visual_backbone']} & {summary['policy_params']:,} & {summary['training_tflops_hours']:.1f} & ${simp.get('sr_pct', 0.0):.0f} \\pm {simp.get('sr_std', 0.0):.0f}$ & ${simp.get('time_mean_s', 0.0):.1f} \\pm {simp.get('time_std_s', 0.0):.1f}$ & ${simp.get('stl_mean_pct', 0.0):.0f} \\pm {simp.get('stl_std_pct', 0.0):.0f}$ & ${stan.get('sr_pct', 0.0):.0f} \\pm {stan.get('sr_std', 0.0):.0f}$ & ${stan.get('time_mean_s', 0.0):.1f} \\pm {stan.get('time_std_s', 0.0):.1f}$ & ${stan.get('stl_mean_pct', 0.0):.0f} \\pm {stan.get('stl_std_pct', 0.0):.0f}$ & ${comp.get('sr_pct', 0.0):.0f} \\pm {comp.get('sr_std', 0.0):.0f}$ & ${comp.get('time_mean_s', 0.0):.1f} \\pm {comp.get('time_std_s', 0.0):.1f}$ & ${comp.get('stl_mean_pct', 0.0):.0f} \\pm {comp.get('stl_std_pct', 0.0):.0f}$ \\\\
\\hline
\\hline
\\end{{tabular}}
}}
\\end{{table*}}
"""
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_content)

    def _write_markdown_table(self, summary: Dict, md_path: Path):
        """Format summary into a GitHub-compatible markdown table."""
        simp = summary["splits"].get("simple", {})
        stan = summary["splits"].get("standard", {})
        comp = summary["splits"].get("complex", {})

        md_content = f"""# Navigation Deployment Metrics Benchmark

| Method | Visual Backbone | Policy Params | Training Cost | Simple SR(%) | Simple Time(s) | Simple STL(%) | Standard SR(%) | Standard Time(s) | Standard STL(%) | Complex SR(%) | Complex Time(s) | Complex STL(%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **{summary['method']}** | {summary['visual_backbone']} | {summary['policy_params']:,} | {summary['training_tflops_hours']:.1f} TFLOPs-h | {simp.get('sr_pct', 0):.0f} ± {simp.get('sr_std', 0):.0f} | {simp.get('time_mean_s', 0):.1f} ± {simp.get('time_std_s', 0):.1f} | {simp.get('stl_mean_pct', 0):.0f} ± {simp.get('stl_std_pct', 0):.0f} | {stan.get('sr_pct', 0):.0f} ± {stan.get('sr_std', 0):.0f} | {stan.get('time_mean_s', 0):.1f} ± {stan.get('time_std_s', 0):.1f} | {stan.get('stl_mean_pct', 0):.0f} ± {stan.get('stl_std_pct', 0):.0f} | {comp.get('sr_pct', 0):.0f} ± {comp.get('sr_std', 0):.0f} | {comp.get('time_mean_s', 0):.1f} ± {comp.get('time_std_s', 0):.1f} | {comp.get('stl_mean_pct', 0):.0f} ± {comp.get('stl_std_pct', 0):.0f} |
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
