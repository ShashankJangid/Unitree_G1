from __future__ import annotations
import argparse
import sys
import pathlib

root_dir = pathlib.Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from g1_teleop import run_teleop
from g1_train import run_training


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 Humanoid Unified Runner")
    parser.add_argument("mode", nargs="?", default="teleop", choices=["teleop", "train"], help="Operation mode (teleop or train)")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    if args.mode == "teleop":
        run_teleop()
    elif args.mode == "train":
        run_training(total_timesteps=args.timesteps, headless=args.headless, resume_path=args.resume)


if __name__ == "__main__":
    main()
