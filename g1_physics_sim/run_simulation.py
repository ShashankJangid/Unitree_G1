from __future__ import annotations
import argparse
import pathlib
import sys
import time
from typing import List, Optional, Tuple
import numpy as np
import mujoco
from mujoco import viewer as mj_viewer

ROBOT_IP = '192.168.123.99'

try:
    from .g1_constants import (    OFFICIAL_KP, OFFICIAL_KD, NOMINAL_STAND_Q, NUM_BODY_MOTORS,
    JOINT_NAMES_29DOF, L_SHOULDER_PITCH, L_WRIST_YAW, R_SHOULDER_PITCH, R_WRIST_YAW,
    LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, LEG_JOINTS, WAIST_JOINTS,
    MODE_PR, DDS_DOMAIN_SIM, DDS_DOMAIN_REAL)
except ImportError:
    from g1_constants import (OFFICIAL_KP, OFFICIAL_KD, NOMINAL_STAND_Q, NUM_BODY_MOTORS,
JOINT_NAMES_29DOF, L_SHOULDER_PITCH, L_WRIST_YAW, R_SHOULDER_PITCH, R_WRIST_YAW,
LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, LEG_JOINTS, WAIST_JOINTS,
MODE_PR, DDS_DOMAIN_SIM, DDS_DOMAIN_REAL)
try:
    from .g1_robot import (    G1Robot)
except ImportError:
    from g1_robot import (G1Robot)
try:
    from .g1_hardware_bridge import (    create_g1_bridge,
    G1RealHardwareBridge)
except ImportError:
    from g1_hardware_bridge import (create_g1_bridge,
G1RealHardwareBridge)
try:
    from .g1_trajectory_manager import (    TrajectoryRecorder,
    TrajectoryPlayer,
    G1MovementScripts)
except ImportError:
    from g1_trajectory_manager import (TrajectoryRecorder,
TrajectoryPlayer,
G1MovementScripts)
try:
    from .g1_policy import (    UnitreeG1Robot)
except ImportError:
    from g1_policy import (UnitreeG1Robot)
try:
    from .g1_policies import (    StandPolicy,
    AnkleSwingPolicy,
    WavePolicy,
    ArmSwingPolicy,
    JointPositionPolicy)
except ImportError:
    from g1_policies import (StandPolicy,
AnkleSwingPolicy,
WavePolicy,
JointPositionPolicy)
try:
    from .g1_state import (    G1State)
except ImportError:
    from g1_state import (G1State)
try:
    from .g1_safety import (    G1FullSafetyManager,
    SafetyState)
except ImportError:
    from g1_safety import (G1FullSafetyManager,
SafetyState)


class G1CartesianController:
    MAX_REACH_RADIUS = 0.40
    MIN_REACH_RADIUS = 0.10

    def __init__(self, robot: G1Robot, right_arm: bool = False):
        self.robot = robot
        self.right_arm = right_arm
        self.left_shoulder_bid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, 'left_shoulder_pitch_link')
        self.right_shoulder_bid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, 'right_shoulder_pitch_link')
        self.joint_limits_min = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
        self.joint_limits_max = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
        for i, jname in enumerate(JOINT_NAMES_29DOF):
            jid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid != -1 and robot.model.jnt_limited[jid]:
                self.joint_limits_min[i] = float(robot.model.jnt_range[jid][0])
                self.joint_limits_max[i] = float(robot.model.jnt_range[jid][1])
            else:
                self.joint_limits_min[i] = -np.pi
                self.joint_limits_max[i] = np.pi
        self.set_arm(right_arm)
        self.target_q = NOMINAL_STAND_Q.copy()

    def set_arm(self, right_arm: bool):
        self.right_arm = right_arm
        if right_arm:
            self.arm_indices = RIGHT_ARM_JOINTS
            self.idle_indices = LEFT_ARM_JOINTS
            self.ee_bid = self.robot.right_hand_bid
            self.shoulder_bid = self.right_shoulder_bid
        else:
            self.arm_indices = LEFT_ARM_JOINTS
            self.idle_indices = RIGHT_ARM_JOINTS
            self.ee_bid = self.robot.left_hand_bid
            self.shoulder_bid = self.left_shoulder_bid

    def clamp_to_safe_workspace(self, raw_goal: np.ndarray) -> Tuple[np.ndarray, bool]:
        shoulder_pos = np.array(self.robot.data.xpos[self.shoulder_bid], dtype=np.float32)
        vec = raw_goal - shoulder_pos
        dist = float(np.linalg.norm(vec))
        was_clamped = False
        if dist < 0.0001:
            vec = np.array([0.2, 0.05 if not self.right_arm else -0.05, -0.15], dtype=np.float32)
            dist = float(np.linalg.norm(vec))
        if dist > self.MAX_REACH_RADIUS:
            vec = vec * (self.MAX_REACH_RADIUS / dist)
            was_clamped = True
        elif dist < self.MIN_REACH_RADIUS:
            vec = vec * (self.MIN_REACH_RADIUS / dist)
            was_clamped = True
        if not self.right_arm:
            if vec[1] < 0.01:
                vec[1] = 0.01
                was_clamped = True
        elif vec[1] > -0.01:
            vec[1] = -0.01
            was_clamped = True
        safe_goal = shoulder_pos + vec
        return (safe_goal, was_clamped)

    def step_ik(self, raw_goal: np.ndarray, damping: float = 0.0001, cart_gain: float = 0.8, nullspace_gain: float = 0.02) -> Tuple[np.ndarray, bool]:
        safe_goal, was_clamped = self.clamp_to_safe_workspace(raw_goal)
        arm_dofs = [self.robot._dof_adr[i] for i in self.arm_indices]
        q_cur_arm = self.target_q[self.arm_indices].copy()

        for _ in range(20):
            cur_p = self.robot.get_hand_pos(self.right_arm)
            err = safe_goal - cur_p
            if np.linalg.norm(err) < 0.0001:
                break
            jacp = np.zeros((3, self.robot.model.nv))
            mujoco.mj_jacBody(self.robot.model, self.robot.data, jacp, None, self.ee_bid)
            J_arm = jacp[:, arm_dofs]
            JJT = J_arm @ J_arm.T + (damping ** 2) * np.eye(3)
            J_pinv = J_arm.T @ np.linalg.solve(JJT, np.eye(3))
            dx = np.clip(err * cart_gain, -0.08, 0.08)
            dq_task = J_pinv @ dx
            N = np.eye(7) - J_pinv @ J_arm
            q_nom_arm = NOMINAL_STAND_Q[self.arm_indices]
            dq_null_desired = nullspace_gain * (q_nom_arm - q_cur_arm)
            dq_null = N @ dq_null_desired
            dq_total = np.clip(dq_task + dq_null, -0.08, 0.08)
            q_cur_arm = q_cur_arm + dq_total
            safe_min = self.joint_limits_min[self.arm_indices] + 0.02
            safe_max = self.joint_limits_max[self.arm_indices] - 0.02
            q_cur_arm = np.clip(q_cur_arm, safe_min, safe_max)
            for k, adr in enumerate([self.robot._qpos_adr[i] for i in self.arm_indices]):
                self.robot.data.qpos[adr] = q_cur_arm[k]
            mujoco.mj_forward(self.robot.model, self.robot.data)

        self.target_q[self.arm_indices] = q_cur_arm
        self.target_q[self.idle_indices] = NOMINAL_STAND_Q[self.idle_indices]
        return (safe_goal, was_clamped)


def main():
    parser = argparse.ArgumentParser(description='Unitree G1 Simulation, Hardware Control & Trajectory Engine')
    parser.add_argument('--hardware', action='store_true', help='Send commands to physical G1 robot via DDS')
    parser.add_argument('--iface', default='Ethernet', help='Network interface for Unitree DDS (e.g. Ethernet, eth0)')
    parser.add_argument('--domain', type=int, default=1, help='DDS domain ID (1=sim, 0=real)')
    parser.add_argument('--free-standing', action='store_true', help='Unlatch gantry harness and balance on ground')
    parser.add_argument('--right-arm', action='store_true', help='Start controlling right arm instead of left')
    parser.add_argument('--record', help='Filename to automatically record trajectory')
    parser.add_argument('--play', help='Trajectory JSON file to play back on startup')
    parser.add_argument('--loop', action='store_true', help='Loop trajectory playback continuously')
    parser.add_argument('--script', choices=['wave', 'circle', 'square', 'standup', 'squat'], help='Run pre-programmed choreography script')
    parser.add_argument('--policy', choices=['stand', 'ankle_swing', 'wave', 'joint_pos'], help='Run a policy from g1_policies')
    parser.add_argument('--duration', type=float, default=0.0, help='Duration for policy/script (0=infinite)')
    args = parser.parse_args()

    fix_base = not args.free_standing
    use_hardware = args.hardware
    domain_id = 0 if use_hardware else args.domain

    print('=' * 72)
    print('   UNITREE G1 REAL-PHYSICS CONTROL, TRAJECTORY RECORDER & SCRIPTING')
    print('=' * 72)
    print(f" Mode:           {('PHYSICAL HARDWARE (DDS)' if use_hardware else 'MUJOCO SIMULATION')}")
    print(f" DDS Domain:     {domain_id}")
    print(f" Stance Mode:    {('GANTRY / SUSPENDED' if fix_base else 'FREE-STANDING')}")
    print(f" Active Arm:     {('RIGHT' if args.right_arm else 'LEFT')}")
    print(f' Reach Limits:   Min 14 cm | Max 34 cm (Physical limit: 38.4 cm)')
    print('-' * 72)
    print(' Interactive Controls:')
    print('   [Mouse Cursor] : Ctrl + Right-Click Drag on green sphere to move arm in 3D')
    print('   W/S/A/D/Q/E    : Step Cartesian Target (Up/Down/Left/Right/Forward/Back)')
    print('   M              : Start / Stop Motion Recording (Saves all 29 motors to JSON)')
    print('   P              : Play last recorded trajectory / loaded motion')
    print('   C              : Execute choreography script (Cycle: Wave -> Circle -> Square)')
    print('   O / T          : Toggle Live Hardware Motor Streaming ON/OFF')
    print('   Space          : Emergency Stop / Disengage Hardware Output')
    print('   1 / 2          : Switch between Left Arm and Right Arm')
    print('   R              : Reset to Safe Nominal Posture')
    print('   ] / [          : Increase / Decrease Step Size')
    print('   Esc            : Quit')
    print('=' * 72)

    robot = G1Robot(fix_base=fix_base)
    bridge = create_g1_bridge(mode='hardware' if use_hardware else 'sim',
                              iface=args.iface, domain_id=domain_id, robot=robot)
    controller = G1CartesianController(robot, right_arm=args.right_arm)
    recorder = TrajectoryRecorder()
    player: Optional[TrajectoryPlayer] = None
    last_saved_file: Optional[pathlib.Path] = None

    policy_runner = None
    if args.policy:
        pol = None
        if args.policy == 'stand':
            pol = StandPolicy()
        elif args.policy == 'ankle_swing':
            pol = AnkleSwingPolicy(duration=args.duration or 3.0)
        elif args.policy == 'wave':
            pol = WavePolicy(right_arm=args.right_arm)
        elif args.policy == 'joint_pos':
            pol = JointPositionPolicy(NOMINAL_STAND_Q)
        elif args.policy == 'arm_swing':
            pol = ArmSwingPolicy(right_arm=args.right_arm)
        if pol is not None:
            policy_runner = pol
            print(f'[POLICY] Driving {args.policy}; interactive Cartesian control is overridden.')

    if args.play:
        try:
            player = TrajectoryPlayer(args.play)
            player.start_playback(loop=args.loop)
        except Exception as e:
            print(f'[ERROR] Could not load trajectory {args.play!r}: {e}')

    if use_hardware and isinstance(bridge, G1RealHardwareBridge):
        if not bridge.wait_for_robot(timeout=5.0):
            print('[FATAL] No LowState on DDS. Nothing will be published.')
            return 4
        problems = bridge.validate_state()
        if problems:
            print('[FATAL] Pre-flight failed; nothing will be published:')
            for p in problems:
                print(f'  - {p}')
            return 5
        real_q, _ = bridge.get_joint_states()
        controller.target_q[:] = real_q[:NUM_BODY_MOTORS]
        robot.reset(real_q[:NUM_BODY_MOTORS])
        print(f'[SYNC] Synchronised with the physical G1. Robot IP is {ROBOT_IP}; '
              'your PC must be a different 192.168.123.x address.')
        if policy_runner is not None and not bridge.start_streaming():
            print('[FATAL] start_streaming() refused.')
            return 6

    target_pos = robot.get_hand_pos(controller.right_arm).copy()
    last_mocap_pos = target_pos.copy()
    robot.set_target_marker(target_pos)
    safety_mgr = G1FullSafetyManager(controller.target_q)
    safety_mgr.set_state(SafetyState.DISARMED)
    safety_mgr.feed_watchdog()

    policy_rate = 50
    policy_t0 = 0.0
    t_start = time.time()
    policy_kp = OFFICIAL_KP.copy()
    policy_kd = OFFICIAL_KD.copy()
    action = G1Action()
    if policy_runner is not None:
        policy_runner.reset(G1State(timestamp=0.0,
                                    q=robot.get_joint_positions(),
                                    dq=robot.get_joint_velocities()))
    active_script_pts: Optional[List[np.ndarray]] = None
    script_pt_idx = 0
    choreography_mode = 0
    step_size = 0.015

    if args.record:
        recorder.start()

    if args.script == 'wave':
        wave_data = G1MovementScripts.create_wave_script(duration=4.0, right_arm=args.right_arm)
        tmp_path = pathlib.Path('trajectories') / 'temp_wave.json'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(wave_data, f)
        player = TrajectoryPlayer(tmp_path)
        player.start_playback(loop=args.loop)
    elif args.script == 'circle':
        active_script_pts = G1MovementScripts.create_circle_cartesian_script(right_arm=args.right_arm)
        script_pt_idx = 0
    elif args.script == 'square':
        active_script_pts = G1MovementScripts.create_square_cartesian_script(right_arm=args.right_arm)
        script_pt_idx = 0
    elif args.script == 'standup':
        standup_data = G1MovementScripts.create_standup_script(duration=3.0)
        tmp_path = pathlib.Path('trajectories') / 'temp_standup.json'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(standup_data, f)
        player = TrajectoryPlayer(tmp_path)
        player.start_playback(loop=False)
    elif args.script == 'squat':
        squat_data = G1MovementScripts.create_squat_script(duration=2.0)
        tmp_path = pathlib.Path('trajectories') / 'temp_squat.json'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(squat_data, f)
        player = TrajectoryPlayer(tmp_path)
        player.start_playback(loop=False)

    def process_key_action(key_char: str):
        nonlocal target_pos, step_size, player, last_saved_file, active_script_pts, script_pt_idx, choreography_mode
        k = key_char.lower()
        if k in ('w', 'up'):
            target_pos[2] += step_size
        elif k in ('s', 'down'):
            target_pos[2] -= step_size
        elif k in ('a', 'left'):
            target_pos[1] += step_size
        elif k in ('d', 'right'):
            target_pos[1] -= step_size
        elif k == 'q':
            target_pos[0] += step_size
        elif k == 'e':
            target_pos[0] -= step_size
        elif k == 'm':
            if not recorder.is_recording:
                recorder.start()
            else:
                saved = recorder.stop_and_save()
                if saved is not None:
                    last_saved_file = saved
        elif k == 'p':
            if player is not None and player.is_playing:
                player.stop()
                print('\n[PLAYER] Playback paused.')
            else:
                play_file = last_saved_file or (pathlib.Path(args.play) if args.play else None)
                if play_file is None:
                    all_json = sorted(list(pathlib.Path('trajectories').glob('*.json')))
                    if len(all_json) > 0:
                        play_file = all_json[-1]
                if play_file is not None and play_file.exists():
                    player = TrajectoryPlayer(play_file)
                    player.start_playback(loop=args.loop)
                else:
                    print("\n[PLAYER] No recorded trajectory file found to play. Press 'M' to record first.")
        elif k == 'c':
            choreography_mode = (choreography_mode + 1) % 3
            if choreography_mode == 0:
                print('\n[CHOREOGRAPHY] >>> Running FRIENDLY WAVE GESTURE <<<')
                wave_data = G1MovementScripts.create_wave_script(duration=4.0, right_arm=controller.right_arm)
                tmp_p = pathlib.Path('trajectories') / 'temp_wave.json'
                with open(tmp_p, 'w', encoding='utf-8') as f:
                    import json
                    json.dump(wave_data, f)
                player = TrajectoryPlayer(tmp_p)
                player.start_playback(loop=False)
            elif choreography_mode == 1:
                print('\n[CHOREOGRAPHY] >>> Tracing CARTESIAN CIRCLE in Y-Z plane <<<')
                active_script_pts = G1MovementScripts.create_circle_cartesian_script(right_arm=controller.right_arm)
                script_pt_idx = 0
            elif choreography_mode == 2:
                print('\n[CHOREOGRAPHY] >>> Tracing CARTESIAN SQUARE in Y-Z plane <<<')
                active_script_pts = G1MovementScripts.create_square_cartesian_script(right_arm=controller.right_arm)
                script_pt_idx = 0
        elif k in ('o', 't'):
            if use_hardware and isinstance(bridge, G1RealHardwareBridge):
                bridge.toggle_streaming()
            else:
                print('\n[INFO] Running in simulation mode (Hardware output disabled).')
        elif k == ' ':
            safety_mgr.emergency_stop('Spacebar Emergency Stop Triggered')
            if use_hardware and isinstance(bridge, G1RealHardwareBridge):
                if bridge.is_streaming:
                    bridge.toggle_streaming()
                    print('\n[E-STOP] Hardware streaming disengaged immediately!')
            if player is not None:
                player.stop()
            active_script_pts = None
            print('\n[SAFETY] >>> EMERGENCY STOP ACTIVE! Joint targets frozen. <<<')
        elif k == '1':
            controller.set_arm(False)
            target_pos = robot.get_hand_pos(False).copy()
            print('\n[ACTION] Switched control to LEFT ARM')
        elif k == '2':
            controller.set_arm(True)
            target_pos = robot.get_hand_pos(True).copy()
            print('\n[ACTION] Switched control to RIGHT ARM')
        elif k == 'r':
            safety_mgr.set_state(SafetyState.ACTIVE_TELEOP)
            safety_mgr.feed_watchdog()
            controller.target_q[:] = NOMINAL_STAND_Q
            target_pos = robot.get_hand_pos(controller.right_arm).copy()
            if player is not None:
                player.stop()
            active_script_pts = None
            print('\n[ACTION] Reset target and posture to nominal safe stand (Safety: ACTIVE_TELEOP)')
        elif k == ']':
            step_size = min(0.08, step_size + 0.005)
            print(f'\n[ACTION] Step size increased to {step_size * 100:.1f} cm')
        elif k == '[':
            step_size = max(0.005, step_size - 0.005)
            print(f'\n[ACTION] Step size decreased to {step_size * 100:.1f} cm')

    def on_viewer_key(keycode: int):
        try:
            if keycode == 32:
                process_key_action(' ')
            elif 32 < keycode <= 126:
                process_key_action(chr(keycode))
            elif keycode == 265:
                process_key_action('up')
            elif keycode == 264:
                process_key_action('down')
            elif keycode == 263:
                process_key_action('left')
            elif keycode == 262:
                process_key_action('right')
        except Exception:
            pass

    try:
        import msvcrt
        def check_console_keys():
            while msvcrt.kbhit():
                ch = msvcrt.getch()
                try:
                    process_key_action(ch.decode('utf-8', errors='ignore'))
                except Exception:
                    pass
    except ImportError:
        def check_console_keys():
            pass

    with mj_viewer.launch_passive(robot.model, robot.data, key_callback=on_viewer_key) as viewer:
        last_print_time = time.time()
        last_hardware_time = time.time()
        last_policy_time = time.time()

        while viewer.is_running():
            step_start = time.time()
            check_console_keys()
            current_mocap = np.array(robot.data.mocap_pos[robot.target_mid], dtype=np.float32)
            if np.linalg.norm(current_mocap - last_mocap_pos) > 0.0001:
                target_pos = current_mocap.copy()

            is_playing_trajectory = False


            if player is not None and player.is_playing:
                q_frame, _, _, _, is_done = player.get_frame()
                if q_frame is not None:
                    controller.target_q[:] = q_frame[:]
                    is_playing_trajectory = True
                if is_done:
                    print('\n[PLAYER] Trajectory playback completed.')
            elif active_script_pts is not None and script_pt_idx < len(active_script_pts):
                target_pos = active_script_pts[script_pt_idx]
                script_pt_idx += 1
                if script_pt_idx >= len(active_script_pts):
                    print('\n[SCRIPT] Script movement choreography completed.')
                    active_script_pts = None


            if policy_runner is not None:
                if policy_runner.get_info().get('t') is None or \
                        (time.time() - policy_t0) >= (1.0 / policy_rate):
                    policy_t0 = time.time()
                    st = G1State(
                        timestamp=time.time() - t_start,
                        q=robot.get_joint_positions(),
                        dq=robot.get_joint_velocities(),
                        tau_est=robot.get_joint_torques(),
                        base_pos=robot.get_base_pose()[0],
                        base_quat=robot.get_base_pose()[1],
                        left_wrist_pos=robot.get_hand_pos(False),
                        right_wrist_pos=robot.get_hand_pos(True),
                        imu_angvel=robot.get_imu_data()[1],
                        imu_linacc=robot.get_imu_data()[2],
                        left_foot_contact=robot.get_foot_contacts()[0],
                        right_foot_contact=robot.get_foot_contacts()[1],
                    )
                    pol_action = policy_runner.step(st)
                    controller.target_q[:] = pol_action.q_des[:NUM_BODY_MOTORS]
                    policy_kp = pol_action.kp
                    policy_kd = pol_action.kd
                    action.mode_pr = pol_action.mode_pr
                    robot.set_mode_pr(pol_action.mode_pr == MODE_PR)
                is_playing_trajectory = True

            if not is_playing_trajectory:
                safe_target, was_clamped = controller.step_ik(target_pos)
                target_pos = safe_target.copy()
            else:
                safe_target = robot.get_hand_pos(controller.right_arm)
                was_clamped = False

            robot.set_target_marker(safe_target)
            last_mocap_pos = safe_target.copy()

            if recorder.is_recording:
                state = G1State(
                    timestamp=time.time(),
                    q=robot.get_joint_positions(),
                    dq=robot.get_joint_velocities(),
                    tau_est=robot.get_joint_torques(),
                    base_pos=robot.get_base_pose()[0],
                    base_quat=robot.get_base_pose()[1],
                    left_wrist_pos=robot.get_hand_pos(False),
                    right_wrist_pos=robot.get_hand_pos(True),
                    imu_angvel=robot.get_imu_data()[1],
                    imu_linacc=robot.get_imu_data()[2],
                    left_foot_contact=robot.get_foot_contacts()[0],
                    right_foot_contact=robot.get_foot_contacts()[1],
                )
                action = G1Action()
                action.q_des[:] = controller.target_q
                action.kp[:] = policy_kp
                action.kd[:] = policy_kd
                recorder.record_state(state, action, safe_target)

            safety_mgr.feed_watchdog()
            guarded, safety_status = safety_mgr.filter_and_guard(
                target_q=controller.target_q,
                current_q=robot.get_joint_positions(),
                current_torques=robot.get_joint_torques(),
                left_wrist_p=robot.get_hand_pos(False),
                right_wrist_p=robot.get_hand_pos(True),
                pelvis_p=robot.get_base_pose()[0],
                base_height=float(robot.get_base_pose()[0][2]),
            )
            controller.target_q[:] = guarded.q_des[:NUM_BODY_MOTORS]
            robot.kp[:] = guarded.kp[:NUM_BODY_MOTORS]
            robot.kd[:] = guarded.kd[:NUM_BODY_MOTORS]


            for _ in range(10):
                robot.step_physics(controller.target_q)


            if use_hardware and time.time() - last_hardware_time >= 0.02:
                bridge.send_joint_targets(controller.target_q)
                last_hardware_time = time.time()

            viewer.sync()

            if time.time() - last_print_time >= 0.4:
                hand_p = robot.get_hand_pos(controller.right_arm)
                err = float(np.linalg.norm(safe_target - hand_p)) * 100.0
                sh_p = robot.data.xpos[controller.shoulder_bid]
                reach_len = float(np.linalg.norm(hand_p - sh_p)) * 100.0
                arm_str = 'RIGHT' if controller.right_arm else 'LEFT '
                clamp_tag = ' [LIMIT!]' if was_clamped else '        '
                rec_tag = ' [REC: ON]' if recorder.is_recording else ''
                play_tag = ' [PLAYING]' if player is not None and player.is_playing else ''
                script_tag = ' [SCRIPT]' if active_script_pts is not None else ''
                hw_status = ' [HW: ACTIVE ]' if bridge.is_hardware_active() else ' [HW: STANDBY]' if use_hardware else ' [SIMULATION]'
                policy_tag = f' [POLICY: {args.policy.upper()}]' if policy_runner is not None else ''
                safety_tag = f' [{safety_status.state.value}]'
                sys.stdout.write(f'\r[{arm_str}]{hw_status}{safety_tag}{rec_tag}{play_tag}{script_tag}{policy_tag} Target:[{safe_target[0]:.2f}, {safe_target[1]:.2f}, {safe_target[2]:.2f}]m | Hand:[{hand_p[0]:.2f}, {hand_p[1]:.2f}, {hand_p[2]:.2f}]m | Reach:{reach_len:4.1f}cm | Err:{err:3.1f}cm{clamp_tag}')
                sys.stdout.flush()
                last_print_time = time.time()

            elapsed = time.time() - step_start
            sleep_time = max(0.0, 0.02 - elapsed)
            time.sleep(sleep_time)

    if recorder.is_recording:
        recorder.stop_and_save()
    bridge.close()
    print('\nSimulation session terminated cleanly.')


if __name__ == '__main__':
    main()
