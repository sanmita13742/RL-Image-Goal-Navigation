# MINav

Reimplementation scaffold for MINav — a small offline RL pipeline for image-goal
navigation. Paper: "120 Minutes and a Laptop: Minimalist Image-goal Navigation via
Unsupervised Exploration and Offline RL" (arXiv:2603.26441).

Idea: let a robot wander around a room on its own using pink noise, encode what it
saw with a frozen DINOv3, and train a goal-conditioned policy offline with TD3+BC.
No human labeling, no online interaction, runs on a laptop in under 2 hours.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/test_install.py
```

## Pipeline

```
collect.py -> encode.py -> train.py (uses relabel.py) -> fqe_select.py -> deploy.py
```

1. `collect.py` — drive the robot around with random (pink noise) actions, save frames + actions
2. `encode.py` — run frames through DINOv3, drop blank/featureless goal candidates
3. `train.py` — offline TD3+BC training with hindsight goal relabeling
4. `fqe_select.py` — pick the best checkpoint without testing on the real robot
5. `deploy.py` — run the policy live

See `docs/EXTRACTION.md` for the math (state/goal/reward, TD3+BC, relabeling, FQE)
and `docs/ARCHITECTURE.md` for how the pieces connect.

## Status

Docs + configs are complete. `scripts/*.py` are working stubs — the noise
generation and a dummy TD3+BC step actually run, but the real 
train/encode/deploy loops need to be wired up to your robot and dataset format.
