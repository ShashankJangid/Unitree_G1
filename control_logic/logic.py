from __future__ import annotations
import abc
import enum
import json
import math
import pathlib
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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
        L_HIP_PITCH, L_HIP_ROLL, L_HIP_YAW, L_KNEE, L_ANKLE_PITCH, L_ANKLE_ROLL,
        R_HIP_PITCH, R_HIP_ROLL, R_HIP_YAW, R_KNEE, R_ANKLE_PITCH, R_ANKLE_ROLL,
        WAIST_YAW, WAIST_ROLL, WAIST_PITCH,
        L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW, L_ELBOW, L_WRIST_ROLL, L_WRIST_PITCH, L_WRIST_YAW,
        R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW, R_ELBOW, R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW,
        ARM_SDK_KP, ARM_SDK_KD, MODE_PR, MODE_AB, MOTOR_MODE_ENABLE, MOTOR_MODE_DISABLE,
        DDS_DOMAIN_SIM, DDS_DOMAIN_REAL, TOPIC_LOW_CMD, TOPIC_LOW_STATE, TOPIC_ARM_SDK,
        FALL_HEIGHT_THRESHOLD, MAX_JOINT_VELOCITY_FULL, ARM_SDK_MAX_JOINT_VELOCITY,
        ARM_SDK_WEIGHT_RATE, ARM_SDK_CONTROL_DT, LOWCMD_CONTROL_DT, POLICY_FREQ, POLICY_DT
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
    @dataclass
    class G1Action:
        q_des: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
        dq_des: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
        kp: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
        kd: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
        tau_ff: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
        motor_mode: np.ndarray = field(default_factory=lambda: np.ones(29, dtype=np.int32))
        arm_sdk_weight: float = 1.0

    @dataclass
    class G1State:
        timestamp: float = 0.0
        q: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
        dq: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
        tau_est: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
        base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
        base_quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=np.float32))
        left_wrist_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
        right_wrist_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
        imu_angvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
        imu_linacc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

try:
    from g1_safety import G1FullSafetyManager, SafetyState
except ImportError:
    class SafetyState(enum.Enum):
        DISARMED = 'DISARMED'
        ACTIVE_TELEOP = 'ACTIVE_TELEOP'
        SOFT_STOP = 'SOFT_STOP'
        EMERGENCY_STOP = 'EMERGENCY_STOP'

    class G1FullSafetyManager:
        def __init__(self, initial_q=None):
            self.state = SafetyState.ACTIVE_TELEOP
        def feed_watchdog(self):
            pass
        def set_state(self, s):
            self.state = s
        def filter_and_guard(self, target_q, current_q, current_torques, left_wrist_p, right_wrist_p, pelvis_p, base_height=0.793):
            a = G1Action()
            a.q_des[:] = target_q[:29]
            a.kp[:] = OFFICIAL_KP[:29]
            a.kd[:] = OFFICIAL_KD[:29]
            return a, self

try:
    from g1_locomotion import G1LocomotionEngine
except ImportError:
    G1LocomotionEngine = None

class Policy(abc.ABC):
    @abc.abstractmethod
    def reset(self, state: G1State) -> None: ...

    @abc.abstractmethod
    def step(self, state: G1State) -> G1Action: ...

    @abc.abstractmethod
    def get_observation(self, state: G1State) -> Any: ...

    @property
    @abc.abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def obs_dim(self) -> int: ...

class StandPolicy(Policy):
    def __init__(self, target_q: Optional[np.ndarray] = None, enable_arm_sdk: bool = True):
        self.target_q = np.array(target_q if target_q is not None else NOMINAL_STAND_Q, dtype=np.float32)[:NUM_BODY_MOTORS]
        self.enable_arm_sdk = enable_arm_sdk
        self._action = G1Action()

    def reset(self, state: G1State) -> None:
        self._action = G1Action()

    def step(self, state: G1State) -> G1Action:
        a = G1Action()
        a.q_des[:] = self.target_q
        a.kp[:] = OFFICIAL_KP[:NUM_BODY_MOTORS]
        a.kd[:] = OFFICIAL_KD[:NUM_BODY_MOTORS]
        a.arm_sdk_weight = 1.0 if self.enable_arm_sdk else 0.0
        return a

    def get_observation(self, state: G1State) -> Any:
        return np.concatenate([state.q[:NUM_BODY_MOTORS], state.dq[:NUM_BODY_MOTORS]])

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return NUM_BODY_MOTORS * 2

class WavePolicy(Policy):
    def __init__(self, right_arm: bool = True, wave_freq: float = 1.5, wave_amplitude: float = 0.4):
        self.right_arm = right_arm
        self.wave_freq = wave_freq
        self.wave_amplitude = wave_amplitude
        self.t0 = 0.0
        self.arm_indices = RIGHT_ARM_JOINTS if right_arm else LEFT_ARM_JOINTS

    def reset(self, state: G1State) -> None:
        self.t0 = state.timestamp

    def step(self, state: G1State) -> G1Action:
        t = state.timestamp - self.t0
        a = G1Action()
        a.q_des[:] = NOMINAL_STAND_Q[:NUM_BODY_MOTORS]
        sh_pitch_idx = self.arm_indices[0]
        elbow_idx = self.arm_indices[3]
        wrist_roll_idx = self.arm_indices[4]
        a.q_des[sh_pitch_idx] = -0.6
        a.q_des[elbow_idx] = 1.4
        a.q_des[wrist_roll_idx] = float(self.wave_amplitude * math.sin(2.0 * math.pi * self.wave_freq * t))
        a.kp[:] = OFFICIAL_KP[:NUM_BODY_MOTORS]
        a.kd[:] = OFFICIAL_KD[:NUM_BODY_MOTORS]
        a.arm_sdk_weight = 1.0
        return a

    def get_observation(self, state: G1State) -> Any:
        return np.concatenate([state.q[:NUM_BODY_MOTORS], state.dq[:NUM_BODY_MOTORS]])

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return NUM_BODY_MOTORS * 2

class AnkleSwingPolicy(Policy):
    def __init__(self, right_arm: bool = True, freq: float = 0.5):
        self.freq = freq
        self.t0 = 0.0

    def reset(self, state: G1State) -> None:
        self.t0 = state.timestamp

    def step(self, state: G1State) -> G1Action:
        t = state.timestamp - self.t0
        a = G1Action()
        a.q_des[:] = NOMINAL_STAND_Q[:NUM_BODY_MOTORS]
        pitch = float(0.15 * math.sin(2.0 * math.pi * self.freq * t))
        a.q_des[L_ANKLE_PITCH] += pitch
        a.q_des[R_ANKLE_PITCH] += pitch
        a.kp[:] = OFFICIAL_KP[:NUM_BODY_MOTORS]
        a.kd[:] = OFFICIAL_KD[:NUM_BODY_MOTORS]
        return a

    def get_observation(self, state: G1State) -> Any:
        return np.concatenate([state.q[:NUM_BODY_MOTORS], state.dq[:NUM_BODY_MOTORS]])

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return NUM_BODY_MOTORS * 2

class UnitreeG1RobotManager:
    def __init__(
        self,
        mode: str = 'sim',
        iface: str = 'Ethernet',
        domain_id: Optional[int] = None,
        policy_rate: int = POLICY_FREQ,
        robot: Any = None,
    ):
        self.mode = mode.lower()
        self.policy_rate = int(policy_rate)
        self.policy_dt = 1.0 / self.policy_rate
        self.control_dt = LOWCMD_CONTROL_DT

        if self.mode == 'sim' and domain_id is None:
            domain_id = DDS_DOMAIN_SIM
        elif self.mode == 'hardware' and domain_id is None:
            domain_id = DDS_DOMAIN_REAL

        from g1_hardware_bridge import create_g1_bridge
        self.bridge = create_g1_bridge(mode=self.mode, iface=iface, domain_id=domain_id, robot=robot)
        self._policy: Optional[Policy] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def set_policy(self, policy: Policy) -> None:
        self._policy = policy

    def start(self) -> bool:
        if self._running:
            return True
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self.bridge:
            self.bridge.close()

    def _run_loop(self) -> None:
        while self._running:
            t0 = time.time()
            q, dq = self.bridge.get_joint_states()
            st = G1State(timestamp=t0, q=q, dq=dq)
            if self._policy:
                act = self._policy.step(st)
                self.bridge.send_joint_targets(act.q_des, act.kp, act.kd, act.dq_des, act.tau_ff)
            elapsed = time.time() - t0
            time.sleep(max(0.0, self.policy_dt - elapsed))

def quat_to_euler_zyx(q: np.ndarray) -> Tuple[float, float, float]:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr_cosp, cosr_cosp))
    sinp = 2.0 * (w * y - z * x)
    pitch = math.degrees(math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny_cosp, cosy_cosp))
    return yaw, pitch, roll
