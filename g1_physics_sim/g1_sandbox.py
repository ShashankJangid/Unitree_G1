"""g1_sandbox.py – Master Unitree G1 Full-Working Simulation & Sandbox API.

Usage:
    # 1. Run interactive sandbox directly:
    python g1_sandbox.py

    # 2. Or import and use the master function in any Python script:
    from g1_sandbox import simulate_g1

    # Simple interactive sandbox:
    simulate_g1()

    # Run with a custom script:
    simulate_g1(script="wave")

    # Run with your own programmatic controller callback (do whatever you want!):
    def my_custom_logic(robot, controller, step):
        # Read sensor states, modify target_q, set Cartesian target, or apply forces
        pass

    simulate_g1(callback=my_custom_logic)
"""

from __future__ import annotations
import argparse
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import mujoco
from mujoco import viewer as mj_viewer


base_dir = pathlib.Path(__file__).resolve().parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

try:  # package and flat-script imports both work
    from .g1_constants import (    OFFICIAL_KP,
    OFFICIAL_KD,
    NOMINAL_STAND_Q,
    ZERO_POSE_Q,
    NUM_BODY_MOTORS,
    JOINT_NAMES_29DOF,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    WAIST_JOINTS,
    LEG_JOINTS,
    L_SHOULDER_PITCH,
    L_ELBOW,
    L_WRIST_ROLL,
    R_SHOULDER_PITCH,
    R_ELBOW,
    R_WRIST_ROLL,
    WAIST_YAW,
    WAIST_ROLL,
    WAIST_PITCH,
    L_HIP_PITCH,
    L_HIP_ROLL,
    L_HIP_YAW,
    L_KNEE,
    L_ANKLE_PITCH,
    L_ANKLE_ROLL,
    R_HIP_PITCH,
    R_HIP_ROLL,
    R_HIP_YAW,
    R_KNEE,
    R_ANKLE_PITCH,
    R_ANKLE_ROLL)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (OFFICIAL_KP,
OFFICIAL_KD,
NOMINAL_STAND_Q,
ZERO_POSE_Q,
NUM_BODY_MOTORS,
JOINT_NAMES_29DOF,
LEFT_ARM_JOINTS,
RIGHT_ARM_JOINTS,
WAIST_JOINTS,
LEG_JOINTS,
L_SHOULDER_PITCH,
L_ELBOW,
L_WRIST_ROLL,
R_SHOULDER_PITCH,
R_ELBOW,
R_WRIST_ROLL,
WAIST_YAW,
WAIST_ROLL,
WAIST_PITCH,
L_HIP_PITCH,
L_HIP_ROLL,
L_HIP_YAW,
L_KNEE,
L_ANKLE_PITCH,
L_ANKLE_ROLL,
R_HIP_PITCH,
R_HIP_ROLL,
R_HIP_YAW,
R_KNEE,
R_ANKLE_PITCH,
R_ANKLE_ROLL)
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
    from .g1_trajectory_manager import (    TrajectoryRecorder,
    TrajectoryPlayer,
    G1MovementScripts)
except ImportError:  # pragma: no cover - flat script layout
    from g1_trajectory_manager import (TrajectoryRecorder,
TrajectoryPlayer,
G1MovementScripts)
try:  # package and flat-script imports both work
    from .g1_safety import (    G1FullSafetyManager,
    SafetyState)
except ImportError:  # pragma: no cover - flat script layout
    from g1_safety import (G1FullSafetyManager,
SafetyState)
from run_simulation import G1CartesianController
try:  # package and flat-script imports both work
    from .g1_state import (    G1State,
    G1Action)
except ImportError:  # pragma: no cover - flat script layout
    from g1_state import (G1State,
G1Action)
try:  # package and flat-script imports both work
    from .g1_lerobot_bridge import (    export_to_unitree_lerobot)
except ImportError:  # pragma: no cover - flat script layout
    from g1_lerobot_bridge import (export_to_unitree_lerobot)


def create_punch_script(duration: float = 3.0, fps: int = 50) -> Dict[str, Any]:
    """Rapid Left Jab + Right Cross boxing combination."""
    num_frames = int(duration * fps)
    frames = []
    base_q = NOMINAL_STAND_Q.copy()


    base_q[L_SHOULDER_PITCH] = -0.4
    base_q[L_SHOULDER_PITCH + 1] = 0.3
    base_q[L_ELBOW] = 1.6
    base_q[R_SHOULDER_PITCH] = -0.4
    base_q[R_SHOULDER_PITCH + 1] = -0.3
    base_q[R_ELBOW] = 1.6

    for idx in range(num_frames):
        t = idx / fps
        q = base_q.copy()

        if 0.5 <= t < 1.3:
            phase = (t - 0.5) / 0.8
            ext = math.sin(phase * math.pi)
            q[L_SHOULDER_PITCH] = -0.4 - 0.5 * ext
            q[L_ELBOW] = 1.6 - 1.2 * ext
            q[WAIST_YAW] = 0.25 * ext

        elif 1.5 <= t < 2.5:
            phase = (t - 1.5) / 1.0
            ext = math.sin(phase * math.pi)
            q[R_SHOULDER_PITCH] = -0.4 - 0.6 * ext
            q[R_ELBOW] = 1.6 - 1.3 * ext
            q[WAIST_YAW] = -0.35 * ext

        frames.append({"time": round(t, 4), "joint_q": [round(float(v), 5) for v in q]})

    return {
        "name": "boxing_punch_combo",
        "duration_sec": duration,
        "fps": fps,
        "num_frames": num_frames,
        "joint_names": JOINT_NAMES_29DOF,
        "frames": frames,
    }


def create_bow_script(duration: float = 3.5, fps: int = 50) -> Dict[str, Any]:
    """Respectful Japanese bowing gesture with waist flexion."""
    num_frames = int(duration * fps)
    frames = []
    base_q = NOMINAL_STAND_Q.copy()

    for idx in range(num_frames):
        t = idx / fps
        phase = t / duration
        bow_depth = math.sin(phase * math.pi)
        q = base_q.copy()
        q[WAIST_PITCH] = 0.55 * bow_depth
        q[L_SHOULDER_PITCH] = 0.2 * bow_depth
        q[R_SHOULDER_PITCH] = 0.2 * bow_depth
        q[L_ELBOW] = 0.3 * bow_depth
        q[R_ELBOW] = 0.3 * bow_depth
        frames.append({"time": round(t, 4), "joint_q": [round(float(v), 5) for v in q]})

    return {
        "name": "respectful_bow",
        "duration_sec": duration,
        "fps": fps,
        "num_frames": num_frames,
        "joint_names": JOINT_NAMES_29DOF,
        "frames": frames,
    }


def print_sandbox_help():
    """Print complete interactive keyboard commands cheat sheet."""
    print("""
+--------------------------------------------------------------------------------------+
|                     UNITREE G1 EDU - INTERACTIVE SANDBOX COMMANDS                    |
+--------------------------------------------------------------------------------------+
|  [MODES & HUMANOID CONTROL]                                                          |
|    '1'           : Select LEFT Arm for Cartesian Reaching (W/S/A/D/Q/E)               |
|    '2'           : Select RIGHT Arm for Cartesian Reaching (W/S/A/D/Q/E)              |
|    '3'           : Toggle WAIST Articulation (W/S Pitch, A/D Yaw, Q/E Roll)           |
|    '4'           : Toggle SQUAT & Stand Height Control (W/S Up/Down)                  |
|    '5' or 'C'    : Cycle Choreography Gestures (Wave, Circle, Square, Punch, Bow)     |
|    '6'           : Control LEFT LEG (W/S Hip Pitch, A/D Hip Roll, Q/E Knee & Ankle)   |
|    '7'           : Control RIGHT LEG (W/S Hip Pitch, A/D Hip Roll, Q/E Knee & Ankle)  |
|    '8'           : Control TORSO TWIST & WHOLE-BODY ARTICULATION (W/S/A/D/Q/E)        |
|    '9'           : Toggle AUTONOMOUS ACT NEURAL POLICY (Learned Pick & Place)         |
|    'J'           : Trigger Official G1 Arm Actions (Salute, Hug, Kiss, Clap, Wave)    |
|    'K'           : Toggle Official Hand Grip (G1 Hand SDK Torque Close / Open)        |
|                                                                                      |
|  [3D CARTESIAN & KEYBOARD NUDGING]                                                   |
|    Ctrl + RClick : Mouse 3D Drag & Drop Target Sphere anywhere in space              |
|    'W' / 'S'     : Nudge Z-Axis (Up / Down)                                          |
|    'A' / 'D'     : Nudge Y-Axis (Left / Right)                                       |
|    'Q' / 'E'     : Nudge X-Axis (Forward / Backward)                                 |
|    '[' / ']'     : Decrease / Increase Nudge Step Size                               |
|    'R'           : Reset Robot to Nominal Safe Stand                                 |
|                                                                                      |
|  [PHYSICS & PERTURBATION]                                                            |
|    'F'           : Apply Physical Push Force to Torso (Test Balance & Compliance)    |
|    'G'           : Change Ground Plane (Toggle Partition Checkerboard / Normal Plane)|
|                                                                                      |
|  [RECORDING, PLAYBACK & HARDWARE]                                                    |
|    'M'           : Toggle 50 Hz Motion Recording (Saves JSON + LeRobot data.json)    |
|    'P'           : Replay Most Recently Recorded Motion                              |
|    'O'           : Toggle Hardware DDS Transmission (with official weight ramp)      |
|    'Space'       : EMERGENCY STOP (Locks motors, zero-torque cutoff)                 |
|    'H'           : Display this Help Screen                                          |
+--------------------------------------------------------------------------------------+
""")


def simulate_g1(
    mode: str = "interactive",
    script: Optional[str] = None,
    trajectory_file: Optional[str] = None,
    callback: Optional[Callable[[G1Robot, G1CartesianController, int], None]] = None,
    right_arm: bool = False,
    free_standing: bool = False,
    hardware: bool = False,
    iface: str = "Ethernet",
    record: Optional[str] = None,
    loop: bool = False,
    duration: float = 0.0,
    headless: bool = False,
    normal_plane: bool = False,
) -> Dict[str, Any]:
    print("=" * 72)
    print("         UNITREE G1 EDU – FULL-WORKING PHYSICS SANDBOX ENGINE")
    print("=" * 72)
    print(f" Mode:           {mode.upper()}")
    print(f" Base Support:   {'FREE-STANDING (Ground Balance)' if free_standing else 'GANTRY HARNESS (Suspended Safe)'}")
    print(f" Ground Plane:   {'NORMAL PLANE (Solid)' if normal_plane else 'PARTITION PLANE (Checkerboard)'}")
    print(f" Hardware DDS:   {'ACTIVE (Physical Robot)' if hardware else 'SIMULATION ONLY'}")
    print(f" Active Arm:     {'RIGHT ARM' if right_arm else 'LEFT ARM'}")
    if script:
        print(f" Script:         {script.upper()}")
    print("=" * 72)


    robot = G1Robot(fix_base=not free_standing, num_motors=NUM_BODY_MOTORS)
    if normal_plane:
        robot.set_plane_type(partition=False)
    controller = G1CartesianController(robot, right_arm=right_arm)
    bridge = create_g1_bridge(mode="hardware" if hardware else "sim", iface=iface, robot=robot)
    recorder = TrajectoryRecorder()


    safety_mgr = G1FullSafetyManager(controller.target_q)
    safety_mgr.set_state(SafetyState.ACTIVE_TELEOP)
    safety_mgr.feed_watchdog()


    if hardware and isinstance(bridge, G1RealHardwareBridge):
        if bridge.wait_for_robot(timeout=4.0):
            real_q, _ = bridge.get_joint_states()
            controller.target_q[:] = real_q[:NUM_BODY_MOTORS]
            robot.reset(real_q[:NUM_BODY_MOTORS])
            print("[SYNC] Synchronized with physical G1 joint angles.")
        else:
            print("[WARN] Robot not responding on DDS. Running in safe standby mode.")


    player: Optional[TrajectoryPlayer] = None
    active_script_pts: Optional[List[np.ndarray]] = None
    script_pt_idx = 0

    if trajectory_file:
        p_path = pathlib.Path(trajectory_file)
        if not p_path.exists():
            alt = base_dir / "trajectories" / trajectory_file
            if alt.exists():
                p_path = alt
        if p_path.exists():
            player = TrajectoryPlayer(p_path)
            player.start_playback(loop=loop)
        else:
            print(f"[ERROR] Trajectory file not found: {trajectory_file}")

    if script:
        s_lower = script.lower()
        if s_lower == "wave":
            w_data = G1MovementScripts.create_wave_script(duration=4.0, right_arm=right_arm)
            tmp_p = base_dir / "trajectories" / "temp_wave.json"
            with open(tmp_p, "w", encoding="utf-8") as f:
                json.dump(w_data, f)
            player = TrajectoryPlayer(tmp_p)
            player.start_playback(loop=loop)
        elif s_lower == "circle":
            active_script_pts = G1MovementScripts.create_circle_cartesian_script(right_arm=right_arm)
            script_pt_idx = 0
        elif s_lower == "square":
            active_script_pts = G1MovementScripts.create_square_cartesian_script(right_arm=right_arm)
            script_pt_idx = 0
        elif s_lower == "squat":
            sq_data = G1MovementScripts.create_squat_script(duration=2.5)
            tmp_p = base_dir / "trajectories" / "temp_squat.json"
            with open(tmp_p, "w", encoding="utf-8") as f:
                json.dump(sq_data, f)
            player = TrajectoryPlayer(tmp_p)
            player.start_playback(loop=loop)
        elif s_lower == "punch":
            p_data = create_punch_script(duration=3.0)
            tmp_p = base_dir / "trajectories" / "temp_punch.json"
            with open(tmp_p, "w", encoding="utf-8") as f:
                json.dump(p_data, f)
            player = TrajectoryPlayer(tmp_p)
            player.start_playback(loop=loop)
        elif s_lower == "bow":
            b_data = create_bow_script(duration=3.5)
            tmp_p = base_dir / "trajectories" / "temp_bow.json"
            with open(tmp_p, "w", encoding="utf-8") as f:
                json.dump(b_data, f)
            player = TrajectoryPlayer(tmp_p)
            player.start_playback(loop=loop)

    if record:
        recorder.start()


    target_pos = robot.get_hand_pos(controller.right_arm).copy()
    last_mocap_pos = target_pos.copy()
    robot.set_target_marker(target_pos)
    step_size = 0.015
    active_mode = "ARM"
    squat_elevation = 0.793
    last_saved_traj: Optional[pathlib.Path] = None
    act_runner = None
    act_active = False


    print_sandbox_help()


    def process_key(k: str):
        nonlocal target_pos, step_size, active_mode, squat_elevation, player, active_script_pts, script_pt_idx, last_saved_traj, act_runner, act_active
        k = k.lower()

        if k == 'h':
            print_sandbox_help()

        elif k == '1':
            controller.set_arm(False)
            target_pos = robot.get_hand_pos(False).copy()
            active_mode = "ARM"
            print('\n[MODE] Controlling LEFT ARM in 3D Cartesian Space.')

        elif k == '2':
            controller.set_arm(True)
            target_pos = robot.get_hand_pos(True).copy()
            active_mode = "ARM"
            print('\n[MODE] Controlling RIGHT ARM in 3D Cartesian Space.')

        elif k == '3':
            active_mode = "WAIST"
            print('\n[MODE] Controlling WAIST Orientation (W/S Pitch, A/D Yaw, Q/E Roll).')

        elif k == '4':
            active_mode = "SQUAT"
            print('\n[MODE] Controlling SQUAT & Pelvis Elevation (W/S Up/Down).')

        elif k in ('5', 'c'):

            gestures = ['wave', 'punch', 'bow', 'circle', 'square', 'squat']
            current_idx = getattr(process_key, '_g_idx', 0)
            g_name = gestures[current_idx % len(gestures)]
            process_key._g_idx = current_idx + 1

            print(f"\n[GESTURE] >>> Executing: {g_name.upper()} <<<")
            if g_name == 'wave':
                d = G1MovementScripts.create_wave_script(duration=4.0, right_arm=controller.right_arm)
            elif g_name == 'punch':
                d = create_punch_script(duration=3.0)
            elif g_name == 'bow':
                d = create_bow_script(duration=3.5)
            elif g_name == 'squat':
                d = G1MovementScripts.create_squat_script(duration=2.5)
            elif g_name == 'circle':
                active_script_pts = G1MovementScripts.create_circle_cartesian_script(right_arm=controller.right_arm)
                script_pt_idx = 0
                return
            elif g_name == 'square':
                active_script_pts = G1MovementScripts.create_square_cartesian_script(right_arm=controller.right_arm)
                script_pt_idx = 0
                return

            tmp_p = base_dir / "trajectories" / f"temp_{g_name}.json"
            with open(tmp_p, "w", encoding="utf-8") as f:
                json.dump(d, f)
            player = TrajectoryPlayer(tmp_p)
            player.start_playback(loop=False)


        elif k == '6':
            active_mode = "LEFT_LEG"
            print('\n[MODE 6] Controlling LEFT LEG (W/S Hip Pitch, A/D Hip Roll, Q/E Knee & Ankle).')

        elif k == '7':
            active_mode = "RIGHT_LEG"
            print('\n[MODE 7] Controlling RIGHT LEG (W/S Hip Pitch, A/D Hip Roll, Q/E Knee & Ankle).')

        elif k == '8':
            active_mode = "TORSO_TWIST"
            print('\n[MODE 8] Controlling TORSO TWIST & WHOLE-BODY ARTICULATION.')

        elif k == '9':
            if not act_active:
                if act_runner is None:
                    try:
                        from g1_act_trainer import G1ACTAutonomousRunner
                        weights = base_dir / "models" / "g1_act" / "act_policy_weights.pt"
                        if weights.exists():
                            act_runner = G1ACTAutonomousRunner(str(weights))
                            print(f'\n[ACT POLICY] Loaded neural weights: {weights.name}')
                        else:
                            print(f'\n[ACT POLICY] No weights found at {weights}. Train first with g1_act_trainer.py --train')
                    except Exception as ex:
                        print(f'\n[ACT POLICY] Error loading model: {ex}')
                if act_runner is not None:
                    act_active = True
                    print('\n[ACT POLICY] >>> AUTONOMOUS ACT NEURAL POLICY ENGAGED! Press 9 to stop. <<<')
            else:
                act_active = False
                print('\n[ACT POLICY] >>> Autonomous policy disengaged. Reverted to teleop. <<<')

        elif k in ('w', 'up'):
            if active_mode == "ARM":
                target_pos[2] += step_size
            elif active_mode == "WAIST":
                controller.target_q[WAIST_PITCH] = np.clip(controller.target_q[WAIST_PITCH] + 0.05, -0.4, 0.6)
            elif active_mode == "SQUAT":
                squat_elevation = min(0.85, squat_elevation + 0.02)
                robot.data.qpos[2] = squat_elevation
            elif active_mode == "LEFT_LEG":
                controller.target_q[L_HIP_PITCH] = np.clip(controller.target_q[L_HIP_PITCH] + 0.05, -1.0, 1.0)
            elif active_mode == "RIGHT_LEG":
                controller.target_q[R_HIP_PITCH] = np.clip(controller.target_q[R_HIP_PITCH] + 0.05, -1.0, 1.0)
            elif active_mode == "TORSO_TWIST":
                controller.target_q[WAIST_PITCH] = np.clip(controller.target_q[WAIST_PITCH] + 0.05, -0.4, 0.6)

        elif k in ('s', 'down'):
            if active_mode == "ARM":
                target_pos[2] -= step_size
            elif active_mode == "WAIST":
                controller.target_q[WAIST_PITCH] = np.clip(controller.target_q[WAIST_PITCH] - 0.05, -0.4, 0.6)
            elif active_mode == "SQUAT":
                squat_elevation = max(0.45, squat_elevation - 0.02)
                robot.data.qpos[2] = squat_elevation

                bend = (0.793 - squat_elevation) * 1.8
                controller.target_q[L_KNEE] = max(0.0, bend)
                controller.target_q[R_KNEE] = max(0.0, bend)
                controller.target_q[L_HIP_PITCH] = -0.5 * bend
                controller.target_q[R_HIP_PITCH] = -0.5 * bend
                controller.target_q[L_ANKLE_PITCH] = -0.5 * bend
                controller.target_q[R_ANKLE_PITCH] = -0.5 * bend
            elif active_mode == "LEFT_LEG":
                controller.target_q[L_HIP_PITCH] = np.clip(controller.target_q[L_HIP_PITCH] - 0.05, -1.0, 1.0)
            elif active_mode == "RIGHT_LEG":
                controller.target_q[R_HIP_PITCH] = np.clip(controller.target_q[R_HIP_PITCH] - 0.05, -1.0, 1.0)
            elif active_mode == "TORSO_TWIST":
                controller.target_q[WAIST_PITCH] = np.clip(controller.target_q[WAIST_PITCH] - 0.05, -0.4, 0.6)

        elif k in ('a', 'left'):
            if active_mode == "ARM":
                target_pos[1] += step_size
            elif active_mode == "WAIST":
                controller.target_q[WAIST_YAW] = np.clip(controller.target_q[WAIST_YAW] + 0.08, -1.2, 1.2)
            elif active_mode == "LEFT_LEG":
                controller.target_q[L_HIP_ROLL] = np.clip(controller.target_q[L_HIP_ROLL] + 0.05, -0.4, 0.4)
            elif active_mode == "RIGHT_LEG":
                controller.target_q[R_HIP_ROLL] = np.clip(controller.target_q[R_HIP_ROLL] + 0.05, -0.4, 0.4)
            elif active_mode == "TORSO_TWIST":
                controller.target_q[WAIST_YAW] = np.clip(controller.target_q[WAIST_YAW] + 0.08, -1.2, 1.2)

        elif k in ('d', 'right'):
            if active_mode == "ARM":
                target_pos[1] -= step_size
            elif active_mode == "WAIST":
                controller.target_q[WAIST_YAW] = np.clip(controller.target_q[WAIST_YAW] - 0.08, -1.2, 1.2)
            elif active_mode == "LEFT_LEG":
                controller.target_q[L_HIP_ROLL] = np.clip(controller.target_q[L_HIP_ROLL] - 0.05, -0.4, 0.4)
            elif active_mode == "RIGHT_LEG":
                controller.target_q[R_HIP_ROLL] = np.clip(controller.target_q[R_HIP_ROLL] - 0.05, -0.4, 0.4)
            elif active_mode == "TORSO_TWIST":
                controller.target_q[WAIST_YAW] = np.clip(controller.target_q[WAIST_YAW] - 0.08, -1.2, 1.2)

        elif k == 'q':
            if active_mode == "ARM":
                target_pos[0] += step_size
            elif active_mode == "WAIST":
                controller.target_q[WAIST_ROLL] = np.clip(controller.target_q[WAIST_ROLL] + 0.05, -0.4, 0.4)
            elif active_mode == "LEFT_LEG":
                controller.target_q[L_KNEE] = np.clip(controller.target_q[L_KNEE] + 0.06, 0.0, 1.8)
                controller.target_q[L_ANKLE_PITCH] = -0.5 * controller.target_q[L_KNEE]
            elif active_mode == "RIGHT_LEG":
                controller.target_q[R_KNEE] = np.clip(controller.target_q[R_KNEE] + 0.06, 0.0, 1.8)
                controller.target_q[R_ANKLE_PITCH] = -0.5 * controller.target_q[R_KNEE]
            elif active_mode == "TORSO_TWIST":
                controller.target_q[WAIST_ROLL] = np.clip(controller.target_q[WAIST_ROLL] + 0.05, -0.4, 0.4)

        elif k == 'e':
            if active_mode == "ARM":
                target_pos[0] -= step_size
            elif active_mode == "WAIST":
                controller.target_q[WAIST_ROLL] = np.clip(controller.target_q[WAIST_ROLL] - 0.05, -0.4, 0.4)
            elif active_mode == "LEFT_LEG":
                controller.target_q[L_KNEE] = np.clip(controller.target_q[L_KNEE] - 0.06, 0.0, 1.8)
                controller.target_q[L_ANKLE_PITCH] = -0.5 * controller.target_q[L_KNEE]
            elif active_mode == "RIGHT_LEG":
                controller.target_q[R_KNEE] = np.clip(controller.target_q[R_KNEE] - 0.06, 0.0, 1.8)
                controller.target_q[R_ANKLE_PITCH] = -0.5 * controller.target_q[R_KNEE]
            elif active_mode == "TORSO_TWIST":
                controller.target_q[WAIST_ROLL] = np.clip(controller.target_q[WAIST_ROLL] - 0.05, -0.4, 0.4)


        elif k == 'f':

            robot.data.xfrc_applied[robot.pelvis_bid, 0] = 60.0
            print('\n[PERTURBATION] >>> Applied 60 N physical push force to robot torso! <<<')

        elif k == 'g':
            is_part = robot.toggle_plane_type()
            p_desc = "PARTITION PLANE (Checkerboard)" if is_part else "NORMAL PLANE (Solid Floor)"
            print(f'\n[ENVIRONMENT] Ground plane changed to: {p_desc}')

        elif k == 'j':
            official_actions = [
                ("right hand wave", [0.4, 0.4, 0.2, 0.8, 0.0, 0.0, 0.0]),
                ("right hand salute", [0.8, -0.2, 0.3, 1.4, 0.0, 0.0, 0.0]),
                ("hands up", [1.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]),
                ("hug", [0.4, 0.6, 0.0, 1.2, 0.0, 0.0, 0.0]),
                ("cross arms", [0.5, 0.3, 0.2, 1.5, 0.0, 0.0, 0.0]),
            ]
            act_idx = getattr(process_key, '_act_idx', 0)
            a_name, a_q = official_actions[act_idx % len(official_actions)]
            process_key._act_idx = act_idx + 1
            print(f'\n[OFFICIAL ACTION] >>> Executing: {a_name.upper()} <<<')
            controller.target_q[RIGHT_ARM_JOINTS] = a_q
            if "both" in a_name or a_name in ("hands up", "hug", "cross arms"):
                controller.target_q[LEFT_ARM_JOINTS] = [a_q[0], -a_q[1], -a_q[2], a_q[3], -a_q[4], a_q[5], -a_q[6]]

        elif k == 'k':
            grip_state = getattr(process_key, '_grip_closed', False)
            grip_state = not grip_state
            process_key._grip_closed = grip_state
            tau = 0.3 if grip_state else -0.3
            state_str = "CLOSED (tau = +0.3 Nm)" if grip_state else "OPEN (tau = -0.3 Nm)"
            print(f'\n[HAND SDK] Hand Grip: {state_str}')


        elif k == 'm':
            if not recorder.is_recording:
                recorder.start()
                print('\n[RECORDER] >>> Recording LIVE 50 Hz motion... Press M to stop and save. <<<')
            else:
                saved = recorder.stop_and_save()
                if saved:
                    last_saved_traj = saved
                    print(f'\n[RECORDER] Saved: {saved.name}')

                    try:
                        lerobot_out = base_dir / "datasets" / f"lerobot_{saved.stem}"
                        export_to_unitree_lerobot(str(saved), str(lerobot_out))
                    except Exception as ex:
                        print(f"[WARN] LeRobot export skipped: {ex}")


        elif k == 'p':
            if player and player.is_playing:
                player.stop()
                print('\n[PLAYER] Playback paused.')
            else:
                p_file = last_saved_traj
                if not p_file:
                    all_j = sorted(list((base_dir / "trajectories").glob("*.json")))
                    if all_j:
                        p_file = all_j[-1]
                if p_file and p_file.exists():
                    player = TrajectoryPlayer(p_file)
                    player.start_playback(loop=loop)
                else:
                    print("\n[PLAYER] No trajectory file found. Press 'M' to record first.")


        elif k in ('o', 't'):
            if hardware and isinstance(bridge, G1RealHardwareBridge):
                bridge.toggle_streaming()
            else:
                print('\n[INFO] Simulation mode (DDS hardware streaming disabled).')


        elif k == ' ':
            safety_mgr.emergency_stop("Spacebar Emergency Stop")
            if hardware and isinstance(bridge, G1RealHardwareBridge):
                bridge.stop_streaming()
            if player:
                player.stop()
            active_script_pts = None
            print('\n[SAFETY] >>> EMERGENCY STOP! All joints locked. <<<')


        elif k == 'r':
            robot.reset()
            safety_mgr.reset(NOMINAL_STAND_Q)
            controller.target_q[:] = NOMINAL_STAND_Q
            target_pos = robot.get_hand_pos(controller.right_arm).copy()
            last_mocap_pos = target_pos.copy()
            robot.set_target_marker(target_pos)
            squat_elevation = 0.793
            active_mode = "ARM"
            if player:
                player.stop()
            active_script_pts = None
            print('\n[RESTART] System, physics, and safety fully re-initialized!')

        elif k == ']':
            step_size = min(0.08, step_size + 0.005)
            print(f'\n[ACTION] Step size: {step_size*100:.1f} cm')
        elif k == '[':
            step_size = max(0.005, step_size - 0.005)
            print(f'\n[ACTION] Step size: {step_size*100:.1f} cm')

    def on_viewer_key(keycode: int):
        try:
            ch = chr(keycode).lower()
            process_key(ch)
        except Exception:
            pass


    try:
        import msvcrt
        def check_console():
            while msvcrt.kbhit():
                try:
                    c = msvcrt.getch().decode('utf-8', errors='ignore')
                    process_key(c)
                except Exception:
                    pass
    except ImportError:
        def check_console():
            pass


    step_idx = 0
    start_time = time.time()
    last_hw_time = time.time()
    last_print_time = time.time()

    def run_simulation_step():
        nonlocal step_idx, target_pos, last_mocap_pos, active_script_pts, script_pt_idx, last_hw_time, last_print_time, act_runner, act_active
        step_start = time.time()
        step_idx += 1
        check_console()


        if not headless:
            current_mocap = np.array(robot.data.mocap_pos[robot.target_mid], dtype=np.float32)
            if np.linalg.norm(current_mocap - last_mocap_pos) > 0.0001:
                target_pos = current_mocap.copy()


        is_playing_motion = False
        if act_active and act_runner is not None:
            controller.target_q[:] = act_runner.query_action(controller.target_q)[:NUM_BODY_MOTORS]
            is_playing_motion = True
        elif player and player.is_playing:
            q_frame, _, _, _, is_done = player.get_frame()
            if q_frame is not None:
                controller.target_q[:] = q_frame[:NUM_BODY_MOTORS]
                is_playing_motion = True
            if is_done:
                print('\n[PLAYER] Motion finished.')
                target_pos = robot.get_hand_pos(controller.right_arm).copy()
                last_mocap_pos = target_pos.copy()
                robot.set_target_marker(target_pos)
        elif active_script_pts and script_pt_idx < len(active_script_pts):
            target_pos = active_script_pts[script_pt_idx]
            script_pt_idx += 1
            if script_pt_idx >= len(active_script_pts):
                print('\n[SCRIPT] Movement completed.')
                active_script_pts = None
                target_pos = robot.get_hand_pos(controller.right_arm).copy()
                last_mocap_pos = target_pos.copy()
                robot.set_target_marker(target_pos)


        if not is_playing_motion and active_mode == "ARM":
            safe_target, was_clamped = controller.step_ik(target_pos)
            target_pos = safe_target.copy()
            robot.set_target_marker(target_pos)
            last_mocap_pos = target_pos.copy()
        else:
            safe_target = robot.get_hand_pos(controller.right_arm)
            was_clamped = False
            if is_playing_motion or active_mode != "ARM":
                robot.set_target_marker(safe_target)
                last_mocap_pos = safe_target.copy()


        if callback:
            try:
                callback(robot, controller, step_idx)
            except Exception as ex:
                print(f"\n[CALLBACK ERROR] {ex}")


        if recorder.is_recording:
            state = G1State(
                timestamp=time.time() - recorder.start_time,
                q=robot.get_joint_positions(),
                dq=robot.get_joint_velocities(),
                tau_est=robot.get_joint_torques(),
                base_pos=robot.get_base_pose()[0],
                base_quat=robot.get_base_pose()[1],
                left_wrist_pos=robot.get_hand_pos(False),
                right_wrist_pos=robot.get_hand_pos(True),
                imu_angvel=robot.get_imu_data()[1],
                imu_linacc=robot.get_imu_data()[2],
            )
            action = G1Action()
            action.q_des[:] = controller.target_q
            action.kp[:] = OFFICIAL_KP[:NUM_BODY_MOTORS]
            action.kd[:] = OFFICIAL_KD[:NUM_BODY_MOTORS]
            recorder.record_state(state, action, safe_target)


        safety_mgr.feed_watchdog()
        action_out, safety_status = safety_mgr.filter_and_guard(
            target_q=controller.target_q,
            current_q=robot.get_joint_positions(),
            current_torques=robot.get_joint_torques(),
            left_wrist_p=robot.get_hand_pos(False),
            right_wrist_p=robot.get_hand_pos(True),
            pelvis_p=robot.get_base_pose()[0],
            base_height=float(robot.get_base_pose()[0][2]),
        )
        controller.target_q[:] = action_out.q_des[:]


        for _ in range(10):
            robot.step_physics(controller.target_q)


        robot.data.xfrc_applied[robot.pelvis_bid, :] = 0.0


        if hardware and (time.time() - last_hw_time) >= 0.02:
            bridge.send_joint_targets(controller.target_q)
            last_hw_time = time.time()


        if time.time() - last_print_time >= 0.35:
            hand_p = robot.get_hand_pos(controller.right_arm)
            err_mm = float(np.linalg.norm(safe_target - hand_p)) * 1000.0
            arm_str = "R_ARM" if controller.right_arm else "L_ARM"
            mode_str = f"[{active_mode}:{arm_str}]"
            hw_str = "[HW: STREAMING]" if bridge.is_hardware_active() else "[SIMULATION]"
            safe_str = f"[{safety_status.state.value}]"
            rec_str = " [REC: ON]" if recorder.is_recording else ""
            play_str = " [PLAY]" if player and player.is_playing else ""

            sys.stdout.write(
                f"\r{mode_str}{safe_str}{hw_str}{rec_str}{play_str} "
                f"Target:[{safe_target[0]:.2f}, {safe_target[1]:.2f}, {safe_target[2]:.2f}]m | "
                f"Hand:[{hand_p[0]:.2f}, {hand_p[1]:.2f}, {hand_p[2]:.2f}]m | "
                f"IK Err:{err_mm:4.1f}mm"
            )
            sys.stdout.flush()
            last_print_time = time.time()


        if duration > 0.0 and (time.time() - start_time) >= duration:
            return False


        dt_step = time.time() - step_start
        time.sleep(max(0.0, 0.02 - dt_step))
        return True

    def safe_simulation_step():
        try:
            return run_simulation_step()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"\n[ERROR SHIELD] Caught simulation anomaly: {ex}")
            robot.reset()
            safety_mgr.reset(NOMINAL_STAND_Q)
            controller.target_q[:] = NOMINAL_STAND_Q
            return True

    if headless:
        print("[SANDBOX] Running in headless mode (no GUI viewer).")
        while True:
            if not safe_simulation_step():
                break
    else:
        with mj_viewer.launch_passive(robot.model, robot.data, key_callback=on_viewer_key) as viewer:
            while viewer.is_running():
                if not safe_simulation_step():
                    break
                try:
                    viewer.sync()
                except Exception:
                    pass


    if recorder.is_recording:
        recorder.stop_and_save()
    bridge.close()
    print("\n[SANDBOX] Simulation terminated cleanly.")

    return {
        "status": "success",
        "steps": step_idx,
        "elapsed_time": time.time() - start_time,
        "final_q": robot.get_joint_positions().tolist(),
        "final_hand_pos": robot.get_hand_pos(controller.right_arm).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Unitree G1 EDU Interactive Sandbox Engine")
    parser.add_argument("--script", choices=["wave", "circle", "square", "squat", "punch", "bow", "ankle_swing"], help="Run pre-programmed gesture")
    parser.add_argument("--play", help="Trajectory JSON file to replay")
    parser.add_argument("--right-arm", action="store_true", help="Start controlling right arm")
    parser.add_argument("--free-standing", action="store_true", help="Balance freely on ground without gantry")
    parser.add_argument("--hardware", action="store_true", help="Stream to physical Unitree G1 robot over DDS")
    parser.add_argument("--iface", default="Ethernet", help="Network interface for Unitree DDS")
    parser.add_argument("--record", help="Filename to record session")
    parser.add_argument("--loop", action="store_true", help="Loop playback continuously")
    parser.add_argument("--duration", type=float, default=0.0, help="Run duration in seconds (0=infinite)")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode without GUI viewer")
    parser.add_argument("--normal-plane", action="store_true", help="Start with normal solid ground plane instead of partition checkerboard")
    args = parser.parse_args()

    simulate_g1(
        mode="headless" if args.headless else "interactive",
        script=args.script,
        trajectory_file=args.play,
        right_arm=args.right_arm,
        free_standing=args.free_standing,
        hardware=args.hardware,
        iface=args.iface,
        record=args.record,
        loop=args.loop,
        duration=args.duration,
        headless=args.headless,
        normal_plane=args.normal_plane,
    )


if __name__ == "__main__":
    main()
