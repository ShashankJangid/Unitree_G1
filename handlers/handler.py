from __future__ import annotations
import abc
import json
import math
import os
import pathlib
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

base_sim_dir = pathlib.Path(__file__).resolve().parent.parent / 'g1_physics_sim'
if str(base_sim_dir) not in sys.path:
    sys.path.insert(0, str(base_sim_dir))

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    from g1_constants import (
        NUM_BODY_MOTORS, NOMINAL_STAND_Q, ZERO_POSE_Q,
        OFFICIAL_KP, OFFICIAL_KD, TORQUE_LIMITS,
        JOINT_LIMITS_MIN, JOINT_LIMITS_MAX, JOINT_NAMES_29DOF,
        LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, WAIST_JOINTS, LEG_JOINTS,
        L_SHOULDER_PITCH, L_ELBOW, L_WRIST_ROLL,
        R_SHOULDER_PITCH, R_ELBOW, R_WRIST_ROLL,
        ARM_SDK_KP, ARM_SDK_KD, MODE_PR, MODE_AB, MOTOR_MODE_ENABLE, MOTOR_MODE_DISABLE,
        DDS_DOMAIN_SIM, DDS_DOMAIN_REAL, TOPIC_LOW_CMD, TOPIC_LOW_STATE, TOPIC_ARM_SDK,
        FALL_HEIGHT_THRESHOLD, MAX_JOINT_VELOCITY_FULL, ARM_SDK_MAX_JOINT_VELOCITY,
        ARM_SDK_WEIGHT_RATE, ARM_SDK_CONTROL_DT, LOWCMD_CONTROL_DT
    )
except ImportError:
    NUM_BODY_MOTORS = 29
    NOMINAL_STAND_Q = np.zeros(29, dtype=np.float32)
    OFFICIAL_KP = np.full(29, 60.0, dtype=np.float32)
    OFFICIAL_KD = np.full(29, 1.0, dtype=np.float32)
    TORQUE_LIMITS = np.full(29, 88.0, dtype=np.float32)
    JOINT_LIMITS_MIN = np.full(29, -np.pi, dtype=np.float32)
    JOINT_LIMITS_MAX = np.full(29, np.pi, dtype=np.float32)
    JOINT_NAMES_29DOF = [f'joint_{i}' for i in range(29)]
    LEFT_ARM_JOINTS = list(range(15, 22))
    RIGHT_ARM_JOINTS = list(range(22, 29))
    WAIST_JOINTS = [12, 13, 14]
    LEG_JOINTS = list(range(12))

try:
    from g1_state import G1Action, G1Observation, G1State
except ImportError:
    G1Action = None
    G1Observation = None
    G1State = None

try:
    from g1_robot import G1Robot
except ImportError:
    G1Robot = None

try:
    from g1_hardware_bridge import (
        G1Bridge,
        G1SimBridge,
        G1RealHardwareBridge,
        create_g1_bridge
    )
except ImportError:
    G1Bridge = None
    G1SimBridge = None
    G1RealHardwareBridge = None
    create_g1_bridge = None

try:
    from g1_safety import G1FullSafetyManager, SafetyState
except ImportError:
    G1FullSafetyManager = None
    SafetyState = None

try:
    from run_simulation import G1CartesianController
except ImportError:
    G1CartesianController = None

try:
    from g1_trajectory_manager import TrajectoryRecorder, TrajectoryPlayer, G1MovementScripts
except ImportError:
    TrajectoryRecorder = None
    TrajectoryPlayer = None
    G1MovementScripts = None

try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, IMUState_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    UNITREE_SDK_AVAILABLE = True
except Exception:
    UNITREE_SDK_AVAILABLE = False
    LowCmd_ = LowState_ = IMUState_ = None
    CRC = MotionSwitcherClient = None
    ChannelFactoryInitialize = ChannelPublisher = ChannelSubscriber = None


class ArmMovementHandler:
    MAX_REACH_RADIUS = 0.40
    MIN_REACH_RADIUS = 0.10

    def __init__(self, robot: Any, right_arm: bool = True):
        self.robot = robot
        self.right_arm = right_arm
        if G1CartesianController is not None and robot is not None:
            self._controller = G1CartesianController(robot, right_arm=right_arm)
        else:
            self._controller = None
        self.target_q = NOMINAL_STAND_Q.copy()
        self.set_arm(right_arm)

    def set_arm(self, right_arm: bool):
        self.right_arm = right_arm
        if self._controller is not None:
            self._controller.set_arm(right_arm)
        if right_arm:
            self.arm_indices = RIGHT_ARM_JOINTS
            self.idle_indices = LEFT_ARM_JOINTS
        else:
            self.arm_indices = LEFT_ARM_JOINTS
            self.idle_indices = RIGHT_ARM_JOINTS

    def move_to_cartesian_point(
        self,
        target_pos: np.ndarray,
        damping: float = 0.0001,
        cart_gain: float = 0.8,
        nullspace_gain: float = 0.02
    ) -> Tuple[np.ndarray, bool]:
        if self._controller is not None:
            safe_goal, was_clamped = self._controller.step_ik(
                raw_goal=target_pos,
                damping=damping,
                cart_gain=cart_gain,
                nullspace_gain=nullspace_gain
            )
            self.target_q[:] = self._controller.target_q[:]
            return safe_goal, was_clamped
        return target_pos, False

    def get_hand_position(self) -> np.ndarray:
        if self.robot is not None and hasattr(self.robot, 'get_hand_pos'):
            return self.robot.get_hand_pos(self.right_arm)
        return np.zeros(3, dtype=np.float32)

    def get_hand_orientation(self) -> Tuple[float, float, float]:
        if self.robot is not None and hasattr(self.robot, 'get_hand_quat'):
            hq = self.robot.get_hand_quat(self.right_arm)
            w, x, y, z = float(hq[0]), float(hq[1]), float(hq[2]), float(hq[3])
            sinr_cosp = 2.0 * (w * x + y * z)
            cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
            roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
            sinp = 2.0 * (w * y - z * x)
            pitch = math.degrees(math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp))
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
            return yaw, pitch, roll
        return 0.0, 0.0, 0.0

    def nudge(self, delta: np.ndarray) -> Tuple[np.ndarray, bool]:
        cur = self.get_hand_position()
        return self.move_to_cartesian_point(cur + delta)


class DataTransferHandler:
    def __init__(self, output_dir: Optional[Union[str, pathlib.Path]] = None):
        self.output_dir = pathlib.Path(output_dir) if output_dir else base_sim_dir / 'trajectories'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if TrajectoryRecorder is not None:
            self._recorder = TrajectoryRecorder(str(self.output_dir))
        else:
            self._recorder = None
        self._player: Optional[Any] = None

    def start_recording(self) -> None:
        if self._recorder is not None:
            self._recorder.start()

    def record_state(self, state: Any, action: Optional[Any] = None, target_pos: Optional[np.ndarray] = None) -> None:
        if self._recorder is not None and self._recorder.is_recording:
            self._recorder.record_frame(state, action, target_pos)

    def stop_and_save(self, filename: Optional[str] = None) -> Optional[pathlib.Path]:
        if self._recorder is not None and self._recorder.is_recording:
            return self._recorder.stop_and_save(filename)
        return None

    def load_trajectory(self, file_path: Union[str, pathlib.Path], loop: bool = False) -> Any:
        if TrajectoryPlayer is not None:
            self._player = TrajectoryPlayer(file_path)
            self._player.start_playback(loop=loop)
            return self._player
        return None

    def get_playback_frame(self, speed_mult: float = 1.0) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
        if self._player is not None:
            return self._player.get_frame(speed_mult=speed_mult)
        return None, None, None, None, True

    def export_to_lerobot(self, json_path: Union[str, pathlib.Path], export_dir: Union[str, pathlib.Path]) -> Optional[pathlib.Path]:
        try:
            from g1_lerobot_bridge import export_to_unitree_lerobot
            return export_to_unitree_lerobot(str(json_path), str(export_dir))
        except Exception:
            return None


class UnitreeRobotInterfaceHandler:
    def __init__(
        self,
        mode: str = 'sim',
        iface: str = 'Ethernet',
        domain_id: Optional[int] = None,
        robot: Optional[Any] = None,
        control_path: str = 'full'
    ):
        self.mode = mode.lower()
        self.iface = iface
        self.domain_id = domain_id if domain_id is not None else (DDS_DOMAIN_REAL if mode == 'hardware' else DDS_DOMAIN_SIM)
        self.control_path = control_path
        self.robot = robot

        if create_g1_bridge is not None:
            self.bridge = create_g1_bridge(
                mode=self.mode,
                iface=self.iface,
                domain_id=self.domain_id,
                robot=self.robot,
                control_path=self.control_path
            )
        else:
            self.bridge = None

    def send_joint_targets(
        self,
        q_des: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        dq_des: Optional[np.ndarray] = None,
        tau_ff: Optional[np.ndarray] = None
    ) -> None:
        if self.bridge is not None:
            self.bridge.send_joint_targets(
                q_des=q_des,
                kp=kp,
                kd=kd,
                dq_des=dq_des,
                tau_ff=tau_ff
            )

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.bridge is not None:
            return self.bridge.get_joint_states()
        return np.zeros(NUM_BODY_MOTORS, dtype=np.float32), np.zeros(NUM_BODY_MOTORS, dtype=np.float32)

    def get_imu_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.bridge is not None:
            return self.bridge.get_imu_state()
        return np.array([1, 0, 0, 0], dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    def wait_for_connection(self, timeout: float = 5.0) -> bool:
        if self.bridge is not None and hasattr(self.bridge, 'wait_for_robot'):
            return self.bridge.wait_for_robot(timeout=timeout)
        return True

    def is_hardware_active(self) -> bool:
        if self.bridge is not None:
            return self.bridge.is_hardware_active()
        return False

    def emergency_stop(self) -> None:
        if self.bridge is not None:
            self.bridge.emergency_stop()

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
