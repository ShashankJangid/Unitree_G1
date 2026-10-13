"""
Run G1 policy in MuJoCo simulation.
Usage:
    python run_sim.py --policy stand
    python run_sim.py --policy ankle_swing --right-arm
    python run_sim.py --policy wave --duration 10
    python run_sim.py --policy locomotion --model models/onnx/locomotion.onnx
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from g1_policy import UnitreeG1Robot, run_policy
from g1_policies import (
    StandPolicy, AnkleSwingPolicy, WavePolicy,
    JointPositionPolicy, LocomotionPolicy
)
from g1_constants import NOMINAL_STAND_Q, NUM_BODY_MOTORS
from config_loader import load_config, create_local_config


def create_policy(args, config):
    """Create policy from command line arguments."""
    if args.policy == "stand":
        target_q = NOMINAL_STAND_Q.copy()
        if args.target_pose:

            pass
        return StandPolicy(
            target_q=target_q,
            kp=config.gains.body_kp,
            kd=config.gains.body_kd,
            enable_arm_sdk=args.enable_arm_sdk
        )

    elif args.policy == "ankle_swing":
        return AnkleSwingPolicy(
            duration=args.duration,
            right_arm=args.right_arm
        )

    elif args.policy == "wave":
        return WavePolicy(
            right_arm=args.right_arm,
            duration=args.duration,
            kp=config.gains.body_kp,
            kd=config.gains.body_kd
        )

    elif args.policy == "joint_pos":
        target_q = NOMINAL_STAND_Q.copy()
        if args.target_values:
            vals = [float(x) for x in args.target_values.split(',')]
            target_q[:len(vals)] = vals
        return JointPositionPolicy(
            target_q=target_q,
            kp=config.gains.body_kp,
            kd=config.gains.body_kd,
            ramp_time=args.ramp_time
        )

    elif args.policy == "locomotion":
        return LocomotionPolicy(policy_path=args.model)

    else:
        raise ValueError(f"Unknown policy: {args.policy}")


def main():
    parser = argparse.ArgumentParser(description="Run G1 policy in MuJoCo simulation")
    parser.add_argument("--policy", type=str, default="stand",
                        choices=["stand", "ankle_swing", "wave", "joint_pos", "locomotion"],
                        help="Policy to run")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--create-config", action="store_true", help="Create local config file")


    parser.add_argument("--right-arm", action="store_true", help="Use right arm for arm policies")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration in seconds")
    parser.add_argument("--enable-arm-sdk", action="store_true", default=True, help="Enable arm SDK")
    parser.add_argument("--target-values", type=str, help="Comma-separated joint targets")
    parser.add_argument("--ramp-time", type=float, default=2.0, help="Ramp time for joint_pos policy")
    parser.add_argument("--model", type=str, help="Path to ONNX model for locomotion policy")


    parser.add_argument("--num-motors", type=int, default=29, choices=[29, 43],
                        help="Number of motors (29 body only, 43 with hands)")
    parser.add_argument("--fix-base", action="store_true", default=True, help="Fix base in simulation")
    parser.add_argument("--free-standing", action="store_true", help="Free-standing (no base fix)")
    parser.add_argument("--no-elastic-band", action="store_true", help="Disable elastic band")
    parser.add_argument("--policy-rate", type=int, default=50, help="Policy rate in Hz")

    args = parser.parse_args()

    if args.create_config:
        create_local_config()
        return 0

    config = load_config(Path(args.config) if args.config else None)


    config.robot.num_motors = args.num_motors
    config.robot.fix_base = not args.free_standing
    config.robot.enable_elastic_band = not args.no_elastic_band
    config.control.policy_freq = args.policy_rate

    print("=" * 60)
    print(f"  Unitree G1 Simulation - Policy: {args.policy}")
    print("=" * 60)
    print(f"  Motors: {config.robot.num_motors}")
    print(f"  Fix base: {config.robot.fix_base}")
    print(f"  Elastic band: {config.robot.enable_elastic_band}")
    print(f"  Policy rate: {config.control.policy_freq} Hz")
    print("=" * 60)

    policy = create_policy(args, config)

    try:
        robot = run_policy(
            policy=policy,
            mode="sim",
            num_motors=config.robot.num_motors,
            policy_rate=config.control.policy_freq,
            duration=args.duration if args.duration > 0 else None,
            fix_base=config.robot.fix_base,
            enable_elastic_band=config.robot.enable_elastic_band,
        )
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
    except Exception as e:
        print(f"\n[Main] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
