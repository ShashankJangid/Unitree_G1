"""
Run G1 policy on real robot hardware.
Usage:
    python run_real.py --policy stand --interface eth0
    python run_real.py --policy ankle_swing --interface enp2s0 --right-arm
    python run_real.py --policy wave --interface eth0 --duration 30

SAFETY REQUIREMENTS:
1. Robot MUST be suspended from gantry/strap
2. Wireless controller with E-stop within reach
3. Clear area around robot
4. Battery charged
5. Start with body-only (29 motors), not Dex3 hands (43 motors)
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
from g1_constants import NOMINAL_STAND_Q, NUM_BODY_MOTORS, DDS_DOMAIN_REAL
from config_loader import load_config, create_local_config


SAFETY_WARNING = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ⚠️  SAFETY WARNING  ⚠️                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This script controls a REAL Unitree G1 robot.                              ║
║                                                                              ║
║  BEFORE RUNNING:                                                             ║
║  ✓ Robot is suspended from overhead gantry/strap                            ║
║  ✓ Wireless controller with E-stop is in hand                               ║
║  ✓ Area is clear of people and objects                                      ║
║  ✓ Battery is sufficiently charged                                          ║
║  ✓ Network interface is correct (check with 'ifconfig' or 'ip a')           ║
║  ✓ Robot IP is set to 192.168.123.99 (default)                              ║
║  ✓ DDS domain is 0 (default for real robot)                                 ║
║                                                                              ║
║  EMERGENCY STOP: Press SELECT on wireless controller or Ctrl+C here         ║
║  The robot WILL FALL if not properly suspended!                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def confirm_safety() -> bool:
    print(SAFETY_WARNING)
    response = input("Type 'I UNDERSTAND THE RISKS' to continue: ")
    return response.strip() == "I UNDERSTAND THE RISKS"


def create_policy(args, config):
    """Create policy from command line arguments."""
    if args.policy == "stand":
        return StandPolicy(
            target_q=NOMINAL_STAND_Q.copy(),
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
    parser = argparse.ArgumentParser(description="Run G1 policy on REAL robot hardware")
    parser.add_argument("--policy", type=str, default="stand",
                        choices=["stand", "ankle_swing", "wave", "joint_pos", "locomotion"],
                        help="Policy to run")
    parser.add_argument("--interface", "-i", type=str, required=True,
                        help="Network interface (e.g., eth0, enp2s0)")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--create-config", action="store_true", help="Create local config file")
    parser.add_argument("--domain", type=int, default=DDS_DOMAIN_REAL, help="DDS domain ID")
    parser.add_argument("--skip-safety", action="store_true", help="Skip safety confirmation (DANGEROUS)")


    parser.add_argument("--right-arm", action="store_true", help="Use right arm for arm policies")
    parser.add_argument("--duration", type=float, default=0.0, help="Duration in seconds (0 = infinite)")
    parser.add_argument("--enable-arm-sdk", action="store_true", default=True, help="Enable arm SDK")
    parser.add_argument("--target-values", type=str, help="Comma-separated joint targets")
    parser.add_argument("--ramp-time", type=float, default=2.0, help="Ramp time for joint_pos policy")
    parser.add_argument("--model", type=str, help="Path to ONNX model for locomotion policy")


    parser.add_argument("--num-motors", type=int, default=29, choices=[29, 43],
                        help="Number of motors (29 body only, 43 with hands)")
    parser.add_argument("--policy-rate", type=int, default=50, help="Policy rate in Hz")

    args = parser.parse_args()

    if args.create_config:
        create_local_config()
        return 0

    if not args.skip_safety:
        if not confirm_safety():
            print("[Main] Safety confirmation failed. Exiting.")
            return 1

    config = load_config(Path(args.config) if args.config else None)


    config.robot.num_motors = args.num_motors
    config.control.policy_freq = args.policy_rate
    config.dds.domain_real = args.domain
    config.dds.interface_real = args.interface

    print("=" * 60)
    print(f"  Unitree G1 REAL ROBOT - Policy: {args.policy}")
    print("=" * 60)
    print(f"  Interface: {args.interface}")
    print(f"  DDS Domain: {args.domain}")
    print(f"  Motors: {config.robot.num_motors}")
    print(f"  Policy rate: {config.control.policy_freq} Hz")
    print("=" * 60)

    policy = create_policy(args, config)

    try:
        robot = UnitreeG1Robot(
            mode="hardware",
            iface=args.interface,
            domain_id=args.domain,
            num_motors=config.robot.num_motors,
            policy_rate=config.control.policy_freq,
        )
        robot.set_policy(policy)

        if robot.start(wait_for_robot=True, timeout=config.hardware.wait_for_robot_timeout):
            print("[Main] Robot started successfully. Running policy...")
            print("[Main] Press Ctrl+C or wireless controller SELECT to stop")

            try:
                if args.duration > 0:
                    time.sleep(args.duration)
                else:
                    while robot.is_running():
                        time.sleep(1.0)
            except KeyboardInterrupt:
                print("\n[Main] Interrupted by user")
            finally:
                robot.stop()
                print("[Main] Robot stopped safely")
        else:
            print("[Main] Failed to start robot")
            return 1

    except Exception as e:
        print(f"\n[Main] Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
