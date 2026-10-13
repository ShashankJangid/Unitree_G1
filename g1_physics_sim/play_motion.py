import argparse
import pathlib
import sys
import time
import numpy as np
from mujoco import viewer as mj_viewer
try:  # package and flat-script imports both work
    from .g1_constants import (    NOMINAL_STAND_Q)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (NOMINAL_STAND_Q)
try:  # package and flat-script imports both work
    from .g1_robot import (    G1Robot)
except ImportError:  # pragma: no cover - flat script layout
    from g1_robot import (G1Robot)
try:  # package and flat-script imports both work
    from .g1_hardware_bridge import (    create_g1_bridge,
    G1RealHardwareBridge)
except ImportError:  # pragma: no cover - flat script layout
    from g1_hardware_bridge import (create_g1_bridge,
G1RealHardwareBridge)
try:  # package and flat-script imports both work
    from .g1_trajectory_manager import (    TrajectoryPlayer)
except ImportError:  # pragma: no cover - flat script layout
    from g1_trajectory_manager import (TrajectoryPlayer)

def play_motion():
    parser = argparse.ArgumentParser(description='Replay recorded motion on Unitree G1')
    parser.add_argument('file', help='Path to motion JSON file in trajectories/')
    parser.add_argument('--hardware', action='store_true', help='Send commands to physical G1 robot via DDS')
    parser.add_argument('--iface', default='Ethernet', help='Network interface for Unitree DDS')
    parser.add_argument('--speed', type=float, default=1.0, help='Playback speed multiplier (e.g. 0.5 for half speed)')
    parser.add_argument('--loop', action='store_true', help='Loop playback continuously')
    parser.add_argument('--free-standing', action='store_true', help='Unlatch gantry harness')
    args = parser.parse_args()
    motion_file = pathlib.Path(args.file)
    if not motion_file.exists():
        alt_path = pathlib.Path('trajectories') / args.file
        if alt_path.exists():
            motion_file = alt_path
        else:
            print(f'[ERROR] Motion file not found: {args.file}')
            sys.exit(1)
    print('=' * 68)
    print('           UNITREE G1 TRAJECTORY REPLAY ENGINE')
    print('=' * 68)
    print(f' File:           {motion_file.name}')
    print(f" Mode:           {('PHYSICAL HARDWARE (DDS)' if args.hardware else 'MUJOCO SIMULATION')}")
    print(f' Playback Speed: {args.speed}x | Loop: {args.loop}')
    print('=' * 68)
    robot = G1Robot(fix_base=not args.free_standing)
    bridge = create_g1_bridge(mode='hardware' if args.hardware else 'sim', iface=args.iface, robot=robot)
    player = TrajectoryPlayer(motion_file)
    if args.hardware and isinstance(bridge, G1RealHardwareBridge):
        if bridge.wait_for_robot(timeout=3.0):
            real_q, _ = bridge.get_joint_states()
            robot.reset(real_q)
            bridge.toggle_streaming()
            print('[SYNC] Connected to physical G1 robot. Starting playback in 1 second...')
            time.sleep(1.0)
        else:
            print('[ERROR] Robot handshake failed. Aborting playback.')
            sys.exit(1)
    player.start_playback(loop=args.loop)
    current_q = NOMINAL_STAND_Q.copy()
    with mj_viewer.launch_passive(robot.model, robot.data) as viewer:
        last_hardware_time = time.time()
        last_print_time = time.time()
        while viewer.is_running() and player.is_playing:
            step_start = time.time()
            q_frame, is_done = player.get_frame(speed_mult=args.speed)
            if q_frame is not None:
                current_q[:] = q_frame[:]
            for _ in range(10):
                robot.step_physics(current_q)
            if args.hardware and time.time() - last_hardware_time >= 0.02:
                bridge.send_joint_targets(current_q)
                last_hardware_time = time.time()
            viewer.sync()
            if time.time() - last_print_time >= 0.3:
                elapsed = time.time() - player.play_start_time
                progress = min(100.0, elapsed / max(0.1, player.duration) * 100.0)
                sys.stdout.write(f'\r[REPLAY] Progress: {progress:5.1f}% | Time: {elapsed:4.1f}s / {player.duration:4.1f}s')
                sys.stdout.flush()
                last_print_time = time.time()
            elapsed_step = time.time() - step_start
            time.sleep(max(0.0, 0.02 - elapsed_step))
    bridge.close()
    print('\n[COMPLETE] Playback completed successfully.')
if __name__ == '__main__':
    play_motion()
