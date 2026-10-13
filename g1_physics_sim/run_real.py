#!/usr/bin/env python3
"""
Run a G1 policy on the REAL robot over DDS.

  # arm_sdk path (upper body only, 17 joints, legs never touched)
  python run_real.py --policy arm_swing --path arm --iface eth0

  # full-body path (all 29 motors on rt/lowcmd)
  python run_real.py --policy stand --path full --iface eth0

PRE-FLIGHT (non-negotiable):
  1. Put the G1 in developer/debug mode: hold R2 + L2 for 20 s.
  2. Connect Ethernet; PC static IP in 192.168.123.0/24, robot is 192.168.123.99.
  3. DDS domain 0 for hardware. Never use the sim domain on a live robot.
  4. Release the onboard motion service is handled automatically
     (MotionSwitcherClient.ReleaseMode) - do not also drive it via LocoClient.
  5. Robot suspended from a gantry unless the policy is leg-only-safe.
  6. Wireless controller in hand. SELECT is the hardware E-stop.
  7. Keep your finger on Ctrl+C: it publishes a motor-disable immediately.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:  # package and flat-script imports both work
    from .g1_constants import (    NOMINAL_STAND_Q,
    NUM_BODY_MOTORS,
    DDS_DOMAIN_REAL)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (NOMINAL_STAND_Q,
NUM_BODY_MOTORS,
DDS_DOMAIN_REAL)
try:  # package and flat-script imports both work
    from .g1_policies import (    AnkleSwingPolicy, ArmSwingPolicy, JointPositionPolicy,
    LocomotionPolicy, StandPolicy, WavePolicy)
except ImportError:  # pragma: no cover - flat script layout
    from g1_policies import (AnkleSwingPolicy, ArmSwingPolicy, JointPositionPolicy,
LocomotionPolicy, StandPolicy, WavePolicy)
try:  # package and flat-script imports both work
    from .g1_policy import (    UnitreeG1Robot)
except ImportError:  # pragma: no cover - flat script layout
    from g1_policy import (UnitreeG1Robot)

POLICIES = ('stand', 'arm_swing', 'wave', 'ankle_swing', 'joint_pos', 'locomotion')

BANNER = r"""
==============================================================================
  REAL HARDWARE CONTROL - READ EVERY LINE BEFORE PROCEEDING
==============================================================================
  [ ] G1 in developer mode (R2 + L2 held 20 s)
  [ ] Ethernet up; PC 192.168.123.x / robot 192.168.123.99
  [ ] DDS domain 0  (arg --domain, default 0)
  [ ] No LocoClient / SportMode running in another process - it fights us
  [ ] Gantry fitted, robot suspended
  [ ] Wireless controller in hand, SELECT tested
  [ ] Area clear, battery > 50%
==============================================================================
"""


def confirm() -> bool:
    print(BANNER)
    if sys.stdin.isatty():
        return input("Type 'ARMED' to continue: ").strip() == 'ARMED'
    print('[warn] non-interactive stdin; proceeding because --skip-safety was given')
    return True


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
    ap = argparse.ArgumentParser(description='Run a G1 policy on real hardware')
    ap.add_argument('--policy', default='stand', choices=POLICIES)
    ap.add_argument('--iface', '-i', required=True,
                    help='NIC with 192.168.123.x (e.g. eth0, enp2s0)')
    ap.add_argument('--domain', type=int, default=DDS_DOMAIN_REAL)
    ap.add_argument('--path', default='full', choices=('full', 'arm'),
                    help="full -> rt/lowcmd (29 motors); arm -> rt/arm_sdk (17 joints)")
    ap.add_argument('--right-arm', action='store_true')
    ap.add_argument('--duration', type=float, default=0.0, help='0 = until Ctrl+C')
    ap.add_argument('--swing-duration', type=float, default=3.0)
    ap.add_argument('--ramp-time', type=float, default=2.0)
    ap.add_argument('--target-values')
    ap.add_argument('--model')
    ap.add_argument('--policy-rate', type=int, default=50)
    ap.add_argument('--wait-timeout', type=float, default=5.0)
    ap.add_argument('--skip-safety', action='store_true')
    ap.add_argument('--check-only', action='store_true',
                    help='run pre-flight validation and exit without publishing')
    args = ap.parse_args()

    if not args.check_only and not args.skip_safety and not confirm():
        print('[main] not confirmed; exiting.')
        return 1

    # ankle_swing drives the legs to the zero posture - refuse the arm path for it
    if args.policy == 'ankle_swing' and args.path == 'arm':
        print('[main] ankle_swing needs the full path: rt/arm_sdk cannot command legs.')
        return 2

    policy = build_policy(args)

    runner = UnitreeG1Robot(
        mode='hardware',
        iface=args.iface,
        domain_id=args.domain,
        policy_rate=args.policy_rate,
        control_path=args.path,
    )

    if not runner.bridge.is_connected:
        print(f'[main] DDS not initialised on {args.iface}/{args.domain}. Aborting.')
        return 3

    if not runner.bridge.wait_for_robot(args.wait_timeout):
        print('[main] no LowState. Check power, cable, NIC name, and IP subnet.')
        return 4

    problems = runner.bridge.validate_state()
    if problems:
        print('[main] PRE-FLIGHT FAILED:')
        for p in problems:
            print(f'  - {p}')
        print('[main] nothing was published. Fix the above and retry.')
        return 5

    print('[main] pre-flight OK. LowState is live and CRC-valid.')
    if args.check_only:
        runner.bridge.close()
        return 0

    runner.set_policy(policy)
    if not runner.start(wait_for_robot=False):
        print('[main] failed to start; nothing is being commanded.')
        runner.bridge.close()
        return 6

    print('[main] ENGAGED. Ctrl+C or the controller SELECT to stop.')
    import time
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while runner.is_running():
                time.sleep(0.5)
    except KeyboardInterrupt:
        print('\n[main] Ctrl+C')
    finally:
        runner.stop()
    print('[main] disarmed.')
    return 0


if __name__ == '__main__':
    sys.exit(main())