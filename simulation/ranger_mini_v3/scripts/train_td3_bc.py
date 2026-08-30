import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.rl.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pipeline.yaml")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir).resolve()
    from scripts.pipeline_utils import setup_pipeline_logger, update_pipeline_state
    
    logger = setup_pipeline_logger(run_dir, "training")
    update_pipeline_state(run_dir, "training", "running")

    try:
        import yaml
        with open(ROOT / args.config) as f:
            full_config = yaml.safe_load(f)

        is_smoke = args.smoke or full_config.get("smoke", {}).get("enabled", False)
        
        # Build the exact config dict that Trainer expects, mapped from pipeline config
        train_cfg = full_config["training"].copy()
        train_cfg["seed"] = full_config["run"]["seed"]
        train_cfg["dataset_path"] = str(run_dir / full_config["hindsight"]["output_subdir"])
        train_cfg["output_dir"] = str(run_dir / full_config["training"]["output_subdir"])
        train_cfg["feature_loading"] = "ram" # Or mmap, hardcode ram for speed
        train_cfg["fqe_frequency"] = train_cfg.get("checkpoint_every", 100000)
        train_cfg["checkpoint_frequency"] = train_cfg.get("checkpoint_every", 100000)
        train_cfg["learning_rate_actor"] = train_cfg["lr_actor"]
        train_cfg["learning_rate_critic"] = train_cfg["lr_critic"]

        if is_smoke:
            train_cfg["total_gradient_steps"] = full_config["smoke"]["training_steps"]
            train_cfg["checkpoint_frequency"] = max(1, train_cfg["total_gradient_steps"] // 2)
            train_cfg["fqe_frequency"] = train_cfg["checkpoint_frequency"]
            logger.warning("SMOKE MODE ACTIVE — limits enforced")

        trainer = Trainer(train_cfg, args.device)
        trainer.train()
        
        logger.info("Training phase completed successfully.")
        update_pipeline_state(run_dir, "training", "completed")
        
        # Write PIPELINE_REPRODUCTION_REPORT.md
        report_path = run_dir / "PIPELINE_REPRODUCTION_REPORT.md"
        with open(report_path, "w") as f:
            f.write(f"# MINav Pipeline Reproduction Report\n\n")
            f.write(f"Run ID: {run_dir.name}\n")
            f.write(f"Status: SUCCESS\n")
            
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        update_pipeline_state(run_dir, "training", "failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
