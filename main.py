from __future__ import annotations
import argparse
import sys
import pathlib
import math
import os

root_dir = pathlib.Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

def main():
    parser = argparse.ArgumentParser(
        description="Unitree G1 Humanoid Unified Runner",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="dance",
        choices=["dance", "gangnam", "teleop", "walk", "wander", "wave", "reach", "bow", "punch", "train"],
        help=(
            "Operation mode:\n"
            "  dance  : Full 29-DOF Saturday Night Disco Fever on standard ground!\n"
            "  gangnam: Dedicated pretrained Gangnam Style policy, no gantry (20.3s excerpt)\n"
            "  teleop : Reactive omnidirectional locomotion with official Unitree RL policy\n"
            "  walk   : Commanded gantry-free walking with the verified locomotion checkpoint\n"
            "  wander : Random bounded walking with strong push-recovery checks\n"
            "  wave   : 29-DOF friendly greeting wave\n"
            "  reach  : 29-DOF tabletop block reaching & picking with dexterous hands\n"
            "  bow    : 29-DOF respectful martial bow\n"
            "  punch  : 29-DOF rapid boxing punches\n"
            "  train  : Whole-body bending RL training"
        )
    )
    parser.add_argument("--timesteps", type=int, default=500000, help="Timesteps for RL training")
    parser.add_argument("--headless", action="store_true", help="Run simulation in headless mode without 3D window")
    parser.add_argument("--duration", type=float, default=None, help="Stop after this many simulated seconds (headless: 10s, Gangnam: one 20.3s cycle)")
    parser.add_argument("--push-force", type=float, default=120.0, help="Torso push magnitude for walk/wander tests")
    parser.add_argument("--auto-push", action="store_true", help="Apply one random push at simulated t=3s")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    args = parser.parse_args()
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")
    if args.duration is not None and args.mode == "train":
        parser.error("--duration is not supported for train; use --timesteps")
    if args.headless:
        # MuJoCo imports its default graphics backend even without a viewer.
        # Disable it before importing any simulation module in this process.
        os.environ["MUJOCO_GL"] = "disable"

    if args.mode == "gangnam":
        from g1_gangnam import run_gangnam

        run_gangnam(headless=args.headless, duration=args.duration)
    elif args.mode == "teleop":
        if args.headless:
            # The same verified 12-DoF checkpoint and nominal 0.4 m/s cruise
            # as teleop, without importing its GUI/terminal controller.
            from g1_velocity import run_velocity_demo
            run_velocity_demo(mode="walk", headless=True, duration=args.duration)
        else:
            from g1_teleop import run_teleop
            run_teleop(duration=args.duration)
    elif args.mode in ("walk", "wander"):
        from g1_velocity import run_velocity_demo

        run_velocity_demo(mode=args.mode, headless=args.headless, push_force=args.push_force, auto_push=args.auto_push, duration=args.duration)
    elif args.mode in ("dance", "wave", "reach", "bow", "punch"):
        from g1_expressive import run_expressive_demo
        run_expressive_demo(mode=args.mode, headless=args.headless, duration=args.duration)
    elif args.mode == "train":
        from g1_train import run_training

        run_training(total_timesteps=args.timesteps, headless=args.headless, resume_path=args.resume)


if __name__ == "__main__":
    main()
