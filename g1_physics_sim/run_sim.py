#!/usr/bin/env python3
"""
Run a G1 policy in MuJoCo simulation.

  python run_sim.py --policy stand
  python run_sim.py --policy arm_swing --right-arm
  python run_sim.py --policy wave --duration 10
  python run_sim.py --policy ankle_swing        # gantry demo, mirrors the SDK example
  python run_sim.py --policy locomotion --model models/onnx/policy.onnx
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np

try:  # package and flat-script imports both work
    from .g1_constants import (    NOMINAL_STAND_Q,
    NUM_BODY_MOTORS)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (NOMINAL_STAND_Q,
NUM_BODY_MOTORS)
try:  # package and flat-script imports both work
    from .g1_policies import (    AnkleSwingPolicy, ArmSwingPolicy, JointPositionPolicy,
    LocomotionPolicy, StandPolicy, WavePolicy)
except ImportError:  # pragma: no cover - flat script layout
    from g1_policies import (AnkleSwingPolicy, ArmSwingPolicy, JointPositionPolicy,
LocomotionPolicy, StandPolicy, WavePolicy)
try:  # package and flat-script imports both work
    from .g1_policy import (    run_policy)
except ImportError:  # pragma: no cover - flat script layout
    from g1_policy import (run_policy)

POLICIES = ('stand', 'arm_swing', 'wave', 'ankle_swing', 'joint_pos', 'locomotion')


def build_policy(args):
    if args.policy == 'stand':
        return StandPolicy(ramp_time=args.ramp_time)
    if args.policy == 'arm_swing':
        return ArmSwingPolicy(right_arm=args.right_arm, duration=args.duration)
    if args.policy == 'wave':
        return WavePolicy(right_arm=args.right_arm, duration=args.duration)
    if args.policy == 'ankle_swing':
        return AnkleSwingPolicy(duration=args.swing_duration)
    if args.policy == 'joint_pos':
        target = NOMINAL_STAND_Q.copy()
        if args.target_values:
            vals = [float(x) for x in args.target_values.split(',')]
            if len(vals) > NUM_BODY_MOTORS:
                raise SystemExit(f'--target-values takes at most {NUM_BODY_MOTORS} values')
            target[:len(vals)] = vals
        return JointPositionPolicy(target, ramp_time=args.ramp_time)
    if args.policy == 'locomotion':
        if not args.model:
            raise SystemExit('--policy locomotion requires --model <path.onnx>')
        p = pathlib.Path(args.model)
        if not p.exists():
            raise SystemExit(f'ONNX model not found: {p}')
        return LocomotionPolicy(policy_path=str(p))
    raise SystemExit(f'unknown policy {args.policy}')


def main() -> int:
    ap = argparse.ArgumentParser(description='Run a G1 policy in MuJoCo simulation')
    ap.add_argument('--policy', default='stand', choices=POLICIES)
    ap.add_argument('--right-arm', action='store_true')
    ap.add_argument('--duration', type=float, default=0.0,
                    help='seconds; 0 runs until Ctrl+C')
    ap.add_argument('--swing-duration', type=float, default=3.0,
                    help='per-stage duration for ankle_swing (official default 3s)')
    ap.add_argument('--ramp-time', type=float, default=2.0)
    ap.add_argument('--target-values', help='comma-separated joint targets')
    ap.add_argument('--model', help='ONNX path for locomotion')
    ap.add_argument('--fix-base', action='store_true', default=True)
    ap.add_argument('--free-standing', action='store_true')
    ap.add_argument('--policy-rate', type=int, default=50)
    args = ap.parse_args()

    if args.free_standing and args.fix_base:
        print('[warn] --free-standing overrides --fix-base')

    policy = build_policy(args)

    print('=' * 64)
    print(f'  Unitree G1 SIM  policy={args.policy}  '
          f'policy_rate={args.policy_rate}Hz')
    print(f'  stance={"free-standing" if args.free_standing else "gantry (base fixed)"}')
    print('=' * 64)

    from g1_robot import G1Robot
    robot = G1Robot(fix_base=not args.free_standing, enable_elastic_band=not args.free_standing)

    runner = run_policy(
        policy,
        mode='sim',
        policy_rate=args.policy_rate,
        duration=args.duration if args.duration > 0 else None,
        robot=robot,
    )
    return 0 if runner else 1


if __name__ == '__main__':
    sys.exit(main())