"""
Policy-facing data containers for the Unitree G1.

Body (29 motors, indices 0..28) and dexterous hands (Dex3-1, 7 DOF per hand)
are kept strictly separate. They arrive on different DDS topics with different
message types and must never be concatenated into one flat index space that
gets fed back to LowCmd_.motor_cmd[].

Quaternion convention: (w, x, y, z) - matches MuJoCo xquat and the SDK's
LowState_.imu_state ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:  # package and flat-script imports both work
    from .g1_constants import (    NUM_BODY_MOTORS, DEX3_NUM_MOTORS, DEX3_JOINTS_PER_HAND,
    JOINT_NAMES_29DOF, OFFICIAL_KP, OFFICIAL_KD, NOMINAL_STAND_Q,
    LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, LEG_JOINTS, WAIST_JOINTS)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (NUM_BODY_MOTORS, DEX3_NUM_MOTORS, DEX3_JOINTS_PER_HAND,
JOINT_NAMES_29DOF, OFFICIAL_KP, OFFICIAL_KD, NOMINAL_STAND_Q,
LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, LEG_JOINTS, WAIST_JOINTS)

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@dataclass
class G1State:
    """Everything a policy may read from the robot."""

    timestamp: float = 0.0

    # --- body, 29 motors, always length 29 ---
    q: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BODY_MOTORS, dtype=np.float32))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BODY_MOTORS, dtype=np.float32))
    tau_est: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BODY_MOTORS, dtype=np.float32))
    motor_state: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BODY_MOTORS, dtype=np.int32))

    # --- dexterous hands, separate subsystem, 7 DOF per hand ---
    left_hand_q: Optional[np.ndarray] = None
    left_hand_dq: Optional[np.ndarray] = None
    right_hand_q: Optional[np.ndarray] = None
    right_hand_dq: Optional[np.ndarray] = None

    # --- base / pelvis ---
    base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_quat: np.ndarray = field(default_factory=lambda: IDENTITY_QUAT.copy())
    base_linvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_angvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # --- torso IMU (rt/secondary_imu) ---
    imu_quat: np.ndarray = field(default_factory=lambda: IDENTITY_QUAT.copy())
    imu_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    imu_angvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    imu_linacc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # --- contacts / limits ---
    left_foot_contact: bool = False
    right_foot_contact: bool = False

    # --- end effectors ---
    left_wrist_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    left_wrist_quat: np.ndarray = field(default_factory=lambda: IDENTITY_QUAT.copy())
    right_wrist_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    right_wrist_quat: np.ndarray = field(default_factory=lambda: IDENTITY_QUAT.copy())

    # --- protocol metadata mirrored from the wire ---
    mode_pr: int = 0
    mode_machine: int = 0
    crc_ok: bool = True

    def __post_init__(self) -> None:
        self.q = np.array(self.q, dtype=np.float32).reshape(NUM_BODY_MOTORS)
        self.dq = np.array(self.dq, dtype=np.float32).reshape(NUM_BODY_MOTORS)
        self.tau_est = np.array(self.tau_est, dtype=np.float32).reshape(NUM_BODY_MOTORS)
        self.motor_state = np.array(self.motor_state, dtype=np.int32).reshape(NUM_BODY_MOTORS)

    @property
    def num_body_motors(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def has_hand_state(self) -> bool:
        return self.left_hand_q is not None or self.right_hand_q is not None

    @property
    def any_motor_faulted(self) -> bool:
        return bool(np.any(self.motor_state != 0))

    def get_faulted_motors(self) -> List[int]:
        return [int(i) for i in np.flatnonzero(self.motor_state != 0)]

    def leg_q(self) -> np.ndarray:
        return self.q[LEG_JOINTS]

    def leg_dq(self) -> np.ndarray:
        return self.dq[LEG_JOINTS]

    def arm_q(self, right: bool = False) -> np.ndarray:
        return self.q[RIGHT_ARM_JOINTS if right else LEFT_ARM_JOINTS]

    def waist_q(self) -> np.ndarray:
        return self.q[WAIST_JOINTS]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'joint_names': list(JOINT_NAMES_29DOF),
            'q': self.q.tolist(),
            'dq': self.dq.tolist(),
            'tau_est': self.tau_est.tolist(),
            'motor_state': self.motor_state.tolist(),
            'base_pos': self.base_pos.tolist(),
            'base_quat': self.base_quat.tolist(),
            'base_linvel': self.base_linvel.tolist(),
            'base_angvel': self.base_angvel.tolist(),
            'imu_quat': self.imu_quat.tolist(),
            'imu_rpy': self.imu_rpy.tolist(),
            'imu_angvel': self.imu_angvel.tolist(),
            'imu_linacc': self.imu_linacc.tolist(),
            'left_foot_contact': self.left_foot_contact,
            'right_foot_contact': self.right_foot_contact,
            'left_wrist_pos': self.left_wrist_pos.tolist(),
            'right_wrist_pos': self.right_wrist_pos.tolist(),
            'left_hand_q': None if self.left_hand_q is None else self.left_hand_q.tolist(),
            'right_hand_q': None if self.right_hand_q is None else self.right_hand_q.tolist(),
            'mode_pr': self.mode_pr,
            'mode_machine': self.mode_machine,
            'crc_ok': self.crc_ok,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'G1State':
        def arr(key: str, n: int, dtype=np.float32) -> np.ndarray:
            return np.asarray(d.get(key) or [0] * n, dtype=dtype)

        s = cls(
            timestamp=float(d.get('timestamp', 0.0)),
            q=arr('q', NUM_BODY_MOTORS),
            dq=arr('dq', NUM_BODY_MOTORS),
            tau_est=arr('tau_est', NUM_BODY_MOTORS),
            motor_state=arr('motor_state', NUM_BODY_MOTORS, np.int32),
            base_pos=arr('base_pos', 3),
            base_quat=arr('base_quat', 4),
            base_linvel=arr('base_linvel', 3),
            base_angvel=arr('base_angvel', 3),
            imu_quat=arr('imu_quat', 4),
            imu_rpy=arr('imu_rpy', 3),
            imu_angvel=arr('imu_angvel', 3),
            imu_linacc=arr('imu_linacc', 3),
            left_foot_contact=bool(d.get('left_foot_contact', False)),
            right_foot_contact=bool(d.get('right_foot_contact', False)),
            left_wrist_pos=arr('left_wrist_pos', 3),
            right_wrist_pos=arr('right_wrist_pos', 3),
            mode_pr=int(d.get('mode_pr', 0)),
            mode_machine=int(d.get('mode_machine', 0)),
            crc_ok=bool(d.get('crc_ok', True)),
        )
        if d.get('left_hand_q'):
            s.left_hand_q = arr('left_hand_q', DEX3_JOINTS_PER_HAND)
        if d.get('right_hand_q'):
            s.right_hand_q = arr('right_hand_q', DEX3_JOINTS_PER_HAND)
        return s


@dataclass
class G1Action:
    """
    A command for one control tick.

    q_des/kp/kd/dq_des/tau_ff are the 29 body slots and are written verbatim
    into LowCmd_.motor_cmd[i]. motor_mode[i] maps to LowCmd_.motor_cmd[i].mode
    (0 disable, 1 enable). The arm_sdk weight lives in `arm_sdk_weight` and is
    written to motor_cmd[29].q, ramped - never hard-switched.
    """

    q_des: np.ndarray = field(default_factory=lambda: NOMINAL_STAND_Q.copy())
    dq_des: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BODY_MOTORS, dtype=np.float32))
    kp: np.ndarray = field(default_factory=lambda: OFFICIAL_KP.copy())
    kd: np.ndarray = field(default_factory=lambda: OFFICIAL_KD.copy())
    tau_ff: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BODY_MOTORS, dtype=np.float32))
    motor_mode: np.ndarray = field(default_factory=lambda: np.ones(NUM_BODY_MOTORS, dtype=np.int32))

    mode_pr: int = 0
    arm_sdk_weight: float = 1.0

    # Dexterous hand targets, 7 DOF per hand. Applied via Dex3Client only.
    left_hand_q_des: Optional[np.ndarray] = None
    right_hand_q_des: Optional[np.ndarray] = None

    timestamp: float = 0.0

    def __post_init__(self) -> None:
        # np.asarray does NOT copy when the dtype already matches, so without
        # explicit copies every G1Action would alias the module-level constants
        # NOMINAL_STAND_Q / OFFICIAL_KP / OFFICIAL_KD. Writing one action's
        # q_des would then mutate those globals and every other live action.
        self.q_des = np.array(self.q_des, dtype=np.float32).reshape(NUM_BODY_MOTORS)
        self.dq_des = np.array(self.dq_des, dtype=np.float32).reshape(NUM_BODY_MOTORS)
        self.kp = np.array(self.kp, dtype=np.float32).reshape(NUM_BODY_MOTORS)
        self.kd = np.array(self.kd, dtype=np.float32).reshape(NUM_BODY_MOTORS)
        self.tau_ff = np.array(self.tau_ff, dtype=np.float32).reshape(NUM_BODY_MOTORS)
        self.motor_mode = np.array(self.motor_mode, dtype=np.int32).reshape(NUM_BODY_MOTORS)
        self.arm_sdk_weight = float(np.clip(self.arm_sdk_weight, 0.0, 1.0))

    def with_arm(self, right: bool, q_arm: np.ndarray) -> 'G1Action':
        idx = RIGHT_ARM_JOINTS if right else LEFT_ARM_JOINTS
        q_arm = np.asarray(q_arm, dtype=np.float32)
        if q_arm.shape[0] != len(idx):
            raise ValueError(f'expected {len(idx)} arm joints, got {q_arm.shape[0]}')
        self.q_des[idx] = q_arm
        return self

    def with_waist(self, q_waist: np.ndarray) -> 'G1Action':
        q_waist = np.asarray(q_waist, dtype=np.float32)
        if q_waist.shape[0] != len(WAIST_JOINTS):
            raise ValueError(f'expected {len(WAIST_JOINTS)} waist joints')
        self.q_des[WAIST_JOINTS] = q_waist
        return self

    def set_gains(self, kp: Optional[np.ndarray] = None, kd: Optional[np.ndarray] = None) -> 'G1Action':
        if kp is not None:
            self.kp = np.asarray(kp, dtype=np.float32).reshape(NUM_BODY_MOTORS)
        if kd is not None:
            self.kd = np.asarray(kd, dtype=np.float32).reshape(NUM_BODY_MOTORS)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'joint_names': list(JOINT_NAMES_29DOF),
            'q_des': self.q_des.tolist(),
            'dq_des': self.dq_des.tolist(),
            'kp': self.kp.tolist(),
            'kd': self.kd.tolist(),
            'tau_ff': self.tau_ff.tolist(),
            'motor_mode': self.motor_mode.tolist(),
            'mode_pr': self.mode_pr,
            'arm_sdk_weight': self.arm_sdk_weight,
            'left_hand_q_des': None if self.left_hand_q_des is None else self.left_hand_q_des.tolist(),
            'right_hand_q_des': None if self.right_hand_q_des is None else self.right_hand_q_des.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'G1Action':
        def arr(key: str, n: int, dtype=np.float32) -> np.ndarray:
            return np.asarray(d.get(key) or [0] * n, dtype=dtype)

        a = cls(
            q_des=arr('q_des', NUM_BODY_MOTORS),
            dq_des=arr('dq_des', NUM_BODY_MOTORS),
            kp=arr('kp', NUM_BODY_MOTORS) if d.get('kp') else OFFICIAL_KP.copy(),
            kd=arr('kd', NUM_BODY_MOTORS) if d.get('kd') else OFFICIAL_KD.copy(),
            tau_ff=arr('tau_ff', NUM_BODY_MOTORS),
            motor_mode=arr('motor_mode', NUM_BODY_MOTORS, np.int32) if d.get('motor_mode')
            else np.ones(NUM_BODY_MOTORS, dtype=np.int32),
            mode_pr=int(d.get('mode_pr', 0)),
            arm_sdk_weight=float(d.get('arm_sdk_weight', 1.0)),
            timestamp=float(d.get('timestamp', 0.0)),
        )
        if d.get('left_hand_q_des'):
            a.left_hand_q_des = arr('left_hand_q_des', DEX3_JOINTS_PER_HAND)
        if d.get('right_hand_q_des'):
            a.right_hand_q_des = arr('right_hand_q_des', DEX3_JOINTS_PER_HAND)
        return a

    @classmethod
    def hold(cls) -> 'G1Action':
        """Damped hold: zero kp, low kd, no feedforward. Used for E-stop / release."""
        a = cls()
        a.kp[:] = 0.0
        a.kd[:] = 1.0
        a.tau_ff[:] = 0.0
        a.q_des[:] = 0.0
        a.motor_mode[:] = 0
        a.arm_sdk_weight = 0.0
        return a


@dataclass
class G1Observation:
    """Flat vector view of a state, for RL policies."""

    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BODY_MOTORS, dtype=np.float32))
    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BODY_MOTORS, dtype=np.float32))
    base_linvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_angvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    projected_gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -1.0], dtype=np.float32))
    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    actions: np.ndarray = field(default_factory=lambda: np.zeros(LEG_JOINTS.__len__(), dtype=np.float32))
    left_foot_contact: bool = False
    right_foot_contact: bool = False
    phase: float = 0.0
    timestamp: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.concatenate([
            self.joint_pos, self.joint_vel,
            self.base_linvel, self.base_angvel, self.projected_gravity,
            np.array([float(self.left_foot_contact), float(self.right_foot_contact),
                      self.phase], dtype=np.float32),
            self.command, self.actions,
        ]).astype(np.float32)

    @property
    def dim(self) -> int:
        return NUM_BODY_MOTORS * 2 + 3 + 3 + 3 + 3 + 3 + len(LEG_JOINTS)


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    """(w, x, y, z) from roll-pitch-yaw in radians."""
    r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cr, sr = np.cos(r * .5), np.sin(r * .5)
    cp, sp = np.cos(p * .5), np.sin(p * .5)
    cy, sy = np.cos(y * .5), np.sin(y * .5)
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy], dtype=np.float32)


def quat_to_rpy(quat: np.ndarray) -> np.ndarray:
    """Roll-pitch-yaw (rad) from (w, x, y, z)."""
    w, x, y, z = (float(v) for v in quat)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return np.zeros(3, dtype=np.float32)
    w, x, y, z = w / n, x / n, y / n, z / n
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw], dtype=np.float32)