from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class G1State:
    """Complete robot state for policy consumption."""
    timestamp: float = 0.0


    q: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
    dq: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
    tau_est: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))


    hand_q: Optional[np.ndarray] = None
    hand_dq: Optional[np.ndarray] = None
    hand_tau: Optional[np.ndarray] = None


    base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    base_linvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_angvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


    imu_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    imu_angvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    imu_linacc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


    left_foot_contact: bool = False
    right_foot_contact: bool = False


    left_hand_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    left_hand_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    right_hand_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    right_hand_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))

    def __post_init__(self):
        if self.hand_q is None:
            self.hand_q = np.zeros(14, dtype=np.float32)
        if self.hand_dq is None:
            self.hand_dq = np.zeros(14, dtype=np.float32)
        if self.hand_tau is None:
            self.hand_tau = np.zeros(14, dtype=np.float32)

    @property
    def num_body_motors(self) -> int:
        return len(self.q)

    @property
    def num_hand_motors(self) -> int:
        return len(self.hand_q) if self.hand_q is not None else 0

    @property
    def total_motors(self) -> int:
        return self.num_body_motors + self.num_hand_motors

    def get_full_q(self) -> np.ndarray:
        if self.num_hand_motors > 0:
            return np.concatenate([self.q, self.hand_q])
        return self.q.copy()

    def get_full_dq(self) -> np.ndarray:
        if self.num_hand_motors > 0:
            return np.concatenate([self.dq, self.hand_dq])
        return self.dq.copy()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'q': self.q.tolist(),
            'dq': self.dq.tolist(),
            'tau_est': self.tau_est.tolist(),
            'hand_q': self.hand_q.tolist() if self.hand_q is not None else [],
            'hand_dq': self.hand_dq.tolist() if self.hand_dq is not None else [],
            'hand_tau': self.hand_tau.tolist() if self.hand_tau is not None else [],
            'base_pos': self.base_pos.tolist(),
            'base_quat': self.base_quat.tolist(),
            'base_linvel': self.base_linvel.tolist(),
            'base_angvel': self.base_angvel.tolist(),
            'imu_quat': self.imu_quat.tolist(),
            'imu_angvel': self.imu_angvel.tolist(),
            'imu_linacc': self.imu_linacc.tolist(),
            'left_foot_contact': self.left_foot_contact,
            'right_foot_contact': self.right_foot_contact,
            'left_hand_pos': self.left_hand_pos.tolist(),
            'left_hand_quat': self.left_hand_quat.tolist(),
            'right_hand_pos': self.right_hand_pos.tolist(),
            'right_hand_quat': self.right_hand_quat.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'G1State':
        state = cls(
            timestamp=data.get('timestamp', 0.0),
            q=np.array(data.get('q', []), dtype=np.float32),
            dq=np.array(data.get('dq', []), dtype=np.float32),
            tau_est=np.array(data.get('tau_est', []), dtype=np.float32),
            base_pos=np.array(data.get('base_pos', [0,0,0]), dtype=np.float32),
            base_quat=np.array(data.get('base_quat', [1,0,0,0]), dtype=np.float32),
            base_linvel=np.array(data.get('base_linvel', [0,0,0]), dtype=np.float32),
            base_angvel=np.array(data.get('base_angvel', [0,0,0]), dtype=np.float32),
            imu_quat=np.array(data.get('imu_quat', [1,0,0,0]), dtype=np.float32),
            imu_angvel=np.array(data.get('imu_angvel', [0,0,0]), dtype=np.float32),
            imu_linacc=np.array(data.get('imu_linacc', [0,0,0]), dtype=np.float32),
            left_foot_contact=data.get('left_foot_contact', False),
            right_foot_contact=data.get('right_foot_contact', False),
            left_hand_pos=np.array(data.get('left_hand_pos', [0,0,0]), dtype=np.float32),
            left_hand_quat=np.array(data.get('left_hand_quat', [1,0,0,0]), dtype=np.float32),
            right_hand_pos=np.array(data.get('right_hand_pos', [0,0,0]), dtype=np.float32),
            right_hand_quat=np.array(data.get('right_hand_quat', [1,0,0,0]), dtype=np.float32),
        )
        if 'hand_q' in data and data['hand_q']:
            state.hand_q = np.array(data['hand_q'], dtype=np.float32)
        if 'hand_dq' in data and data['hand_dq']:
            state.hand_dq = np.array(data['hand_dq'], dtype=np.float32)
        if 'hand_tau' in data and data['hand_tau']:
            state.hand_tau = np.array(data['hand_tau'], dtype=np.float32)
        return state


@dataclass
class G1Action:
    """Robot action for policy output."""
    timestamp: float = 0.0


    q_des: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
    dq_des: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
    kp: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
    kd: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
    tau_ff: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))


    hand_q_des: Optional[np.ndarray] = None
    hand_dq_des: Optional[np.ndarray] = None
    hand_kp: Optional[np.ndarray] = None
    hand_kd: Optional[np.ndarray] = None
    hand_tau_ff: Optional[np.ndarray] = None


    motor_modes: Optional[np.ndarray] = None
    hand_modes: Optional[np.ndarray] = None


    enable_arm_sdk: bool = True

    def __post_init__(self):
        if self.hand_q_des is None:
            self.hand_q_des = np.zeros(14, dtype=np.float32)
        if self.hand_dq_des is None:
            self.hand_dq_des = np.zeros(14, dtype=np.float32)
        if self.hand_kp is None:
            self.hand_kp = np.zeros(14, dtype=np.float32)
        if self.hand_kd is None:
            self.hand_kd = np.zeros(14, dtype=np.float32)
        if self.hand_tau_ff is None:
            self.hand_tau_ff = np.zeros(14, dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'q_des': self.q_des.tolist(),
            'dq_des': self.dq_des.tolist(),
            'kp': self.kp.tolist(),
            'kd': self.kd.tolist(),
            'tau_ff': self.tau_ff.tolist(),
            'hand_q_des': self.hand_q_des.tolist() if self.hand_q_des is not None else [],
            'hand_dq_des': self.hand_dq_des.tolist() if self.hand_dq_des is not None else [],
            'hand_kp': self.hand_kp.tolist() if self.hand_kp is not None else [],
            'hand_kd': self.hand_kd.tolist() if self.hand_kd is not None else [],
            'hand_tau_ff': self.hand_tau_ff.tolist() if self.hand_tau_ff is not None else [],
            'motor_modes': self.motor_modes.tolist() if self.motor_modes is not None else [],
            'hand_modes': self.hand_modes.tolist() if self.hand_modes is not None else [],
            'enable_arm_sdk': self.enable_arm_sdk,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'G1Action':
        action = cls(
            timestamp=data.get('timestamp', 0.0),
            q_des=np.array(data.get('q_des', []), dtype=np.float32),
            dq_des=np.array(data.get('dq_des', []), dtype=np.float32),
            kp=np.array(data.get('kp', []), dtype=np.float32),
            kd=np.array(data.get('kd', []), dtype=np.float32),
            tau_ff=np.array(data.get('tau_ff', []), dtype=np.float32),
            enable_arm_sdk=data.get('enable_arm_sdk', True),
        )
        for field_name in ['hand_q_des', 'hand_dq_des', 'hand_kp', 'hand_kd', 'hand_tau_ff']:
            if field_name in data and data[field_name]:
                setattr(action, field_name, np.array(data[field_name], dtype=np.float32))
        if 'motor_modes' in data and data['motor_modes']:
            action.motor_modes = np.array(data['motor_modes'], dtype=np.int32)
        if 'hand_modes' in data and data['hand_modes']:
            action.hand_modes = np.array(data['hand_modes'], dtype=np.int32)
        return action

    @classmethod
    def zero(cls, num_body: int = 29, num_hand: int = 14) -> 'G1Action':
        return cls(
            q_des=np.zeros(num_body, dtype=np.float32),
            dq_des=np.zeros(num_body, dtype=np.float32),
            kp=np.zeros(num_body, dtype=np.float32),
            kd=np.zeros(num_body, dtype=np.float32),
            tau_ff=np.zeros(num_body, dtype=np.float32),
            hand_q_des=np.zeros(num_hand, dtype=np.float32),
            hand_dq_des=np.zeros(num_hand, dtype=np.float32),
            hand_kp=np.zeros(num_hand, dtype=np.float32),
            hand_kd=np.zeros(num_hand, dtype=np.float32),
            hand_tau_ff=np.zeros(num_hand, dtype=np.float32),
        )

    @classmethod
    def nominal_stand(cls, num_body: int = 29, num_hand: int = 14) -> 'G1Action':
        from g1_constants import NOMINAL_STAND_Q, DEFAULT_KP, DEFAULT_KD
        action = cls.zero(num_body, num_hand)
        action.q_des[:] = NOMINAL_STAND_Q[:num_body]
        action.kp[:] = DEFAULT_KP[:num_body]
        action.kd[:] = DEFAULT_KD[:num_body]
        if num_hand > 0:
            action.hand_kp[:] = DEFAULT_KP[num_body:num_body+num_hand]
            action.hand_kd[:] = DEFAULT_KD[num_body:num_body+num_hand]
        return action


@dataclass
class G1Observation:
    """Policy observation - typically a subset of state or processed features."""
    timestamp: float = 0.0


    joint_pos: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
    joint_vel: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))


    base_linvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_angvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))


    imu_angvel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    imu_linacc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


    left_foot_contact: bool = False
    right_foot_contact: bool = False


    command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


    phase: float = 0.0


    extras: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        parts = [
            self.joint_pos,
            self.joint_vel,
            self.base_linvel,
            self.base_angvel,
            self.base_quat,
            self.imu_angvel,
            self.imu_linacc,
            np.array([float(self.left_foot_contact), float(self.right_foot_contact)], dtype=np.float32),
            self.command,
            np.array([self.phase], dtype=np.float32),
        ]
        for v in self.extras.values():
            if isinstance(v, np.ndarray):
                parts.append(v.flatten())
            else:
                parts.append(np.array([v], dtype=np.float32))
        return np.concatenate(parts)

    @property
    def dim(self) -> int:
        return self.to_array().shape[0]
