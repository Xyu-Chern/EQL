"""Launch the archived paired CGPO protocol in its PyTorch/MuJoCo v3 environment."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(task, seed, output):
    source = Path(__file__).resolve().parents[1] / "provenance" / "cgpo_a100_v3" / "src"
    command = [sys.executable, "-u", str(source / "run_paired.py"), "--env", task,
               "--seed", str(seed), "--cuda", "cuda:0", "--aug", "--weighted", "--running_v"]
    subprocess.run(command, cwd=source, env={**os.environ, "CGPO_RESULTS": str(output.resolve())}, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("Hopper-v3", "Walker2d-v3"))
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.task, args.seed, args.output)
