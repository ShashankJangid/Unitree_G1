from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np


L_LEG_HIP_PITCH = 0
L_LEG_HIP_ROLL = 1
L_LEG_HIP_YAW = 2
L_LEG_KNEE = 3
L_LEG_ANKLE_PITCH = 4
L_LEG_ANKLE_ROLL = 5
R_LEG_HIP_PITCH = 6
R_LEG_HIP_ROLL = 7
R_LEG_HIP_YAW = 8
R_LEG_KNEE = 9
R_LEG_ANKLE_PITCH = 10
R_LEG_ANKLE_ROLL = 11
WAIST_YAW = 12
WAIST_ROLL = 13
WAIST_PITCH = 14
L_SHOULDER_PITCH = 15
L_SHOULDER_ROLL = 16
L_SHOULDER_YAW = 17
L_ELBOW = 18
L_WRIST_ROLL = 19
L_WRIST_PITCH = 20
L_WRIST_YAW = 21
R_SHOULDER_PITCH = 22
R_SHOULDER_ROLL = 23
R_SHOULDER_YAW = 24
R_ELBOW = 25
R_WRIST_ROLL = 26
R_WRIST_PITCH = 27
R_WRIST_YAW = 28


ARM_SDK_ENABLE = 29


L_HAND_THUMB_0 = 30
L_HAND_THUMB_1 = 31
L_HAND_THUMB_2 = 32
L_HAND_INDEX_0 = 33
L_HAND_INDEX_1 = 34
L_HAND_MIDDLE_0 = 35
L_HAND_MIDDLE_1 = 36
R_HAND_THUMB_0 = 37
R_HAND_THUMB_1 = 38
R_HAND_THUMB_2 = 39
R_HAND_INDEX_0 = 40
R_HAND_INDEX_1 = 41
R_HAND_MIDDLE_0 = 42
R_HAND_MIDDLE_1 = 43

NUM_BODY_MOTORS = 29
NUM_HAND_MOTORS = 14
NUM_MOTORS = 43

JOINT_NAMES_29DOF: List[str] = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
]

JOINT_NAMES_43DOF: List[str] = JOINT_NAMES_29DOF + [
    'left_hand_thumb_0_joint', 'left_hand_thumb_1_joint', 'left_hand_thumb_2_joint',
    'left_hand_index_0_joint', 'left_hand_index_1_joint',
    'left_hand_middle_0_joint', 'left_hand_middle_1_joint',
    'right_hand_thumb_0_joint', 'right_hand_thumb_1_joint', 'right_hand_thumb_2_joint',
    'right_hand_index_0_joint', 'right_hand_index_1_joint',
    'right_hand_middle_0_joint', 'right_hand_middle_1_joint'
]


DEFAULT_KP_29 = np.array([
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
    100.0, 100.0, 100.0, 150.0, 40.0, 40.0,
    100.0, 60.0, 60.0,
    40.0, 40.0, 40.0, 40.0, 25.0, 25.0, 25.0,
    40.0, 40.0, 40.0, 40.0, 25.0, 25.0, 25.0
], dtype=np.float32)

DEFAULT_KD_29 = np.array([
    3.0, 3.0, 3.0, 4.0, 1.5, 1.5,
    3.0, 3.0, 3.0, 4.0, 1.5, 1.5,
    3.0, 2.0, 2.0,
    1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8,
    1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8
], dtype=np.float32)


TORQUE_LIMITS_29 = np.array([
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 50.0, 50.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0,
    25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0
], dtype=np.float32)


DEFAULT_KP_HAND = np.full(14, 20.0, dtype=np.float32)
DEFAULT_KD_HAND = np.full(14, 0.5, dtype=np.float32)
TORQUE_LIMITS_HAND = np.full(14, 2.0, dtype=np.float32)


DEFAULT_KP = np.concatenate([DEFAULT_KP_29, DEFAULT_KP_HAND])
DEFAULT_KD = np.concatenate([DEFAULT_KD_29, DEFAULT_KD_HAND])
TORQUE_LIMITS = np.concatenate([TORQUE_LIMITS_29, TORQUE_LIMITS_HAND])


NOMINAL_STAND_Q = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
NOMINAL_STAND_Q[L_LEG_HIP_PITCH] = -0.28
NOMINAL_STAND_Q[L_LEG_KNEE] = 0.60
NOMINAL_STAND_Q[L_LEG_ANKLE_PITCH] = -0.32
NOMINAL_STAND_Q[R_LEG_HIP_PITCH] = -0.28
NOMINAL_STAND_Q[R_LEG_KNEE] = 0.60
NOMINAL_STAND_Q[R_LEG_ANKLE_PITCH] = -0.32
NOMINAL_STAND_Q[L_SHOULDER_PITCH] = 0.35
NOMINAL_STAND_Q[L_SHOULDER_ROLL] = 0.18
NOMINAL_STAND_Q[L_ELBOW] = 0.50
NOMINAL_STAND_Q[R_SHOULDER_PITCH] = 0.35
NOMINAL_STAND_Q[R_SHOULDER_ROLL] = -0.18
NOMINAL_STAND_Q[R_ELBOW] = 0.50


ZERO_POSE_Q = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)


JOINT_LIMITS_MIN_29 = np.array([
    -2.7, -0.5, -2.7, -0.1, -0.8, -0.5,
    -2.7, -0.5, -2.7, -0.1, -0.8, -0.5,
    -2.6, -0.5, -0.5,
    -3.0, -1.5, -2.7, -1.0, -2.7, -1.5, -2.7,
    -3.0, -1.5, -2.7, -1.0, -2.7, -1.5, -2.7
], dtype=np.float32)

JOINT_LIMITS_MAX_29 = np.array([
    2.7, 2.7, 2.7, 2.9, 0.8, 0.5,
    2.7, 2.7, 2.7, 2.9, 0.8, 0.5,
    2.6, 0.5, 0.5,
    2.7, 2.7, 2.7, 2.1, 2.7, 1.5, 2.7,
    2.7, 2.7, 2.7, 2.1, 2.7, 1.5, 2.7
], dtype=np.float32)


JOINT_LIMITS_MIN_HAND = np.full(14, -1.0, dtype=np.float32)
JOINT_LIMITS_MAX_HAND = np.full(14, 1.0, dtype=np.float32)

JOINT_LIMITS_MIN = np.concatenate([JOINT_LIMITS_MIN_29, JOINT_LIMITS_MIN_HAND])
JOINT_LIMITS_MAX = np.concatenate([JOINT_LIMITS_MAX_29, JOINT_LIMITS_MAX_HAND])


LEG_JOINT_INDICES = [
    L_LEG_HIP_PITCH, L_LEG_HIP_ROLL, L_LEG_HIP_YAW, L_LEG_KNEE, L_LEG_ANKLE_PITCH, L_LEG_ANKLE_ROLL,
    R_LEG_HIP_PITCH, R_LEG_HIP_ROLL, R_LEG_HIP_YAW, R_LEG_KNEE, R_LEG_ANKLE_PITCH, R_LEG_ANKLE_ROLL
]


LEFT_ARM_INDICES = list(range(L_SHOULDER_PITCH, L_WRIST_YAW + 1))
RIGHT_ARM_INDICES = list(range(R_SHOULDER_PITCH, R_WRIST_YAW + 1))


WAIST_INDICES = [WAIST_YAW, WAIST_ROLL, WAIST_PITCH]


DDS_DOMAIN_SIM = 1
DDS_DOMAIN_REAL = 0
DDS_INTERFACE_LOOPBACK = "lo"


OFFICIAL_TOPIC_ARM_SDK = "rt/arm_sdk"
OFFICIAL_TOPIC_LOW_CMD = "rt/lowcmd"
OFFICIAL_TOPIC_LOW_STATE = "rt/lowstate"
OFFICIAL_NUM_MOTOR_SLOTS = 35
ARM_SDK_WEIGHT_INDEX = 29


OFFICIAL_KP_SHOULDER = 300.0
OFFICIAL_KD_SHOULDER = 3.0
OFFICIAL_KP_ELBOW = 80.0
OFFICIAL_KD_ELBOW = 3.0
OFFICIAL_KP_WRIST = 40.0
OFFICIAL_KD_WRIST = 1.5
OFFICIAL_KP_WAIST = 10000.0
OFFICIAL_KD_WAIST = 10000.0


ISAACLAB_KP_SHOULDER = 300.0
ISAACLAB_KD_SHOULDER = 3.0
ISAACLAB_KP_ELBOW = 400.0
ISAACLAB_KD_ELBOW = 2.5
ISAACLAB_KP_WRIST = 400.0
ISAACLAB_KD_WRIST = 2.5
ISAACLAB_KP_WAIST = 10000.0
ISAACLAB_KD_WAIST = 10000.0


OFFICIAL_ISAAC_STAND_Q = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
OFFICIAL_ISAAC_STAND_Q[L_LEG_HIP_PITCH] = -0.05
OFFICIAL_ISAAC_STAND_Q[L_LEG_KNEE] = 0.20
OFFICIAL_ISAAC_STAND_Q[L_LEG_ANKLE_PITCH] = -0.15
OFFICIAL_ISAAC_STAND_Q[R_LEG_HIP_PITCH] = -0.05
OFFICIAL_ISAAC_STAND_Q[R_LEG_KNEE] = 0.20
OFFICIAL_ISAAC_STAND_Q[R_LEG_ANKLE_PITCH] = -0.15


TOPIC_LOW_CMD = "rt/arm_sdk"
TOPIC_LOW_STATE = "rt/lowstate"
TOPIC_SPORT_MODE_STATE = "rt/sportmodestate"
TOPIC_IMU_STATE = "rt/secondary_imu"
TOPIC_HAND_CMD = "rt/hand_sdk"
TOPIC_HAND_STATE = "rt/handstate"


PHYSICS_FREQ = 500
CONTROL_FREQ = 50
DECIMATION = PHYSICS_FREQ // CONTROL_FREQ


MAX_JOINT_VELOCITY = 1.2
MAX_JOINT_ACCELERATION = 10.0
EMERGENCY_STOP_DECEL = 20.0


ELASTIC_BAND_STIFFNESS = 5000.0
ELASTIC_BAND_DAMPING = 100.0
ELASTIC_BAND_HEIGHT = 0.793

@dataclass
class G1Config:
    num_motors: int = NUM_BODY_MOTORS
    use_hands: bool = False
    fix_base: bool = True
    dds_domain: int = DDS_DOMAIN_SIM
    dds_interface: str = DDS_INTERFACE_LOOPBACK
    control_freq: int = CONTROL_FREQ
    physics_freq: int = PHYSICS_FREQ
    enable_elastic_band: bool = True
    elastic_band_height: float = ELASTIC_BAND_HEIGHT
    kp: np.ndarray = field(default_factory=lambda: DEFAULT_KP.copy())
    kd: np.ndarray = field(default_factory=lambda: DEFAULT_KD.copy())
    torque_limits: np.ndarray = field(default_factory=lambda: TORQUE_LIMITS.copy())
    joint_limits_min: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_MIN.copy())
    joint_limits_max: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_MAX.copy())

    def __post_init__(self):
        if self.num_motors == NUM_BODY_MOTORS:
            self.kp = self.kp[:NUM_BODY_MOTORS].copy()
            self.kd = self.kd[:NUM_BODY_MOTORS].copy()
            self.torque_limits = self.torque_limits[:NUM_BODY_MOTORS].copy()
            self.joint_limits_min = self.joint_limits_min[:NUM_BODY_MOTORS].copy()
            self.joint_limits_max = self.joint_limits_max[:NUM_BODY_MOTORS].copy()
        elif self.num_motors == NUM_MOTORS:
            pass
        else:
            raise ValueError(f"num_motors must be {NUM_BODY_MOTORS} or {NUM_MOTORS}")

ARM_SDK_JOINTS = [
    L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW, L_ELBOW, L_WRIST_ROLL,
    R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW, R_ELBOW, R_WRIST_ROLL,
    WAIST_YAW, WAIST_ROLL, WAIST_PITCH
]


ARM_ACTIONS = {
    "release arm": 99,
    "two-hand kiss": 11,
    "left kiss": 12,
    "right kiss": 13,
    "hands up": 15,
    "clap": 17,
    "high five": 18,
    "hug": 19,
    "right hand wave": 20,
    "right hand wave2": 21,
    "shake hand": 22,
    "right hand salute": 23,
    "left hand wave": 24,
    "left hand wave2": 25,
    "left hand salute": 26,
    "right hand raise": 27,
    "left hand raise": 28,
    "both hands raise": 29,
    "cross arms": 30,
}

LOCO_ACTIONS = {
    "damp": 0,
    "Squat2StandUp": 1,
    "StandUp2Squat": 2,
    "move forward": 3,
    "move lateral": 4,
    "move rotate": 5,
    "low stand": 6,
    "high stand": 7,
    "zero torque": 8,
    "wave hand1": 9,
    "wave hand2": 10,
    "shake hand": 11,
    "Lie2StandUp": 12,
}


MOTOR_MODE_DISABLE = 0
MOTOR_MODE_ENABLE = 1
MOTOR_MODE_POSITION = 1
MOTOR_MODE_VELOCITY = 2
MOTOR_MODE_TORQUE = 3


RIS_MODE_POSITION = 0x01
RIS_MODE_VELOCITY = 0x02
RIS_MODE_TORQUE = 0x04
RIS_MODE_DAMPING = 0x08

def encode_dex3_mode(mode: int, kp: float = 0.0, kd: float = 0.0) -> int:
    """Encode RIS mode byte for Dex3 hand motors."""
    return (mode & 0x0F) | (int(kp * 10) & 0xF) << 4 | (int(kd * 10) & 0xF) << 8

def decode_dex3_mode(mode_byte: int) -> Tuple[int, float, float]:
    """Decode RIS mode byte for Dex3 hand motors."""
    mode = mode_byte & 0x0F
    kp = ((mode_byte >> 4) & 0xF) / 10.0
    kd = ((mode_byte >> 8) & 0xF) / 10.0
    return mode, kp, kd
