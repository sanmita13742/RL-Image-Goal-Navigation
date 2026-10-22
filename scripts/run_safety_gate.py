#!/usr/bin/env python3
"""
Convenience forwarder for onboard robot execution:
  python scripts/run_safety_gate.py
Delegates directly to realworld/scripts/run_safety_gate.py
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "realworld" / "scripts" / "run_safety_gate.py"

if __name__ == "__main__":
    cmd = [sys.executable, str(TARGET)] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))
