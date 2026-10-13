"""
Unitree G1 constants, verified line-by-line against the official SDK.

Sources of truth:
  * official_unitree_repos/unitree_sdk2/example/g1/low_level/g1_ankle_swing_example.cpp
  * official_unitree_repos/unitree_sdk2/example/g1/low_level/g1_dual_arm_example.cpp
  * official_unitree_repos/unitree_sdk2/example/g1/high_level/g1_arm7_sdk_dds_example.cpp
  * official_unitree_repos/unitree_sdk2/example/g1/high_level/g1_hand_sdk_example.cpp
  * g1_physics_sim/models/g1/g1_joint_index_dds.md

CRITICAL FACT: the dexterous hands are NOT part of LowCmd_.motor_cmd[].
LowCmd_/LowState_ carry 29 body motors at indices 0..28 plus unused slots at
29..34. Hands travel on their own topic with their own message type.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# =====================================================================
# BODY JOINT INDICES  (29 motors, 0..28)  - matches SDK G1JointIndex
# =====================================================================
# Left leg (0-5)
L_HIP_PITCH = 0
L_HIP_ROLL = 1
L_HIP_YAW = 2
L_KNEE = 3
L_ANKLE_PITCH = 4
L_ANKLE_ROLL = 5
# Right leg (6-11)
R_HIP_PITCH = 6
R_HIP_ROLL = 7
R_HIP_YAW = 8
R_KNEE = 9
R_ANKLE_PITCH = 10
R_ANKLE_ROLL = 11
# Waist (12-14)
WAIST_YAW = 12
WAIST_ROLL = 13
WAIST_PITCH = 14
# Left arm (15-21)
L_SHOULDER_PITCH = 15
L_SHOULDER_ROLL = 16
L_SHOULDER_YAW = 17
L_ELBOW = 18
L_WRIST_ROLL = 19
L_WRIST_PITCH = 20
L_WRIST_YAW = 21
# Right arm (22-28)
R_SHOULDER_PITCH = 22
R_SHOULDER_ROLL = 23
R_SHOULDER_YAW = 24
R_ELBOW = 25
R_WRIST_ROLL = 26
R_WRIST_PITCH = 27
R_WRIST_YAW = 28

NUM_BODY_MOTORS = 29

# --- ankle PR / AB aliasing (g1_ankle_swing_example.cpp) -------------
# In AB (parallel) mode the SDK re-maps the same two slots:
#   L_ANKLE_B == 4, L_ANKLE_A == 5, R_ANKLE_B == 10, R_ANKLE_A == 11
#   WAIST_A == 13, WAIST_B == 14
L_ANKLE_B = L_ANKLE_PITCH
L_ANKLE_A = L_ANKLE_ROLL
R_ANKLE_B = R_ANKLE_PITCH
R_ANKLE_A = R_ANKLE_ROLL
WAIST_A = WAIST_ROLL
WAIST_B = WAIST_PITCH

# LowCmd_.motor_cmd[] extends past 28 with unused slots.
K_NOT_USED_JOINT = 29
K_NOT_USED_JOINT1 = 30
K_NOT_USED_JOINT2 = 31
K_NOT_USED_JOINT3 = 32
K_NOT_USED_JOINT4 = 33
K_NOT_USED_JOINT5 = 34
ARM_SDK_ENABLE = K_NOT_USED_JOINT  # motor_cmd[29].q : 1=enable, 0=disable

# motor_cmd / motor_state array size in LowCmd_ and LowState_
LOWCMD_SLOTS = 35

# =====================================================================
# DEX3-1 HAND (separate topic + separate message type)
# =====================================================================
# Dex3-1 = 7 DOF per hand, ordered:
#   0 thumb_0, 1 thumb_1, 2 thumb_2, 3 index_0, 4 index_1,
#   5 middle_0, 6 middle_1
DEX3_JOINTS_PER_HAND = 7
DEX3_JOINT_NAMES: List[str] = [
    'thumb_0', 'thumb_1', 'thumb_2',
    'index_0', 'index_1',
    'middle_0', 'middle_1',
]
DEX3_NUM_MOTORS = DEX3_JOINTS_PER_HAND * 2  # both hands

# rt/hand_sdk (g1_hand_sdk_example.cpp) uses a *different*, smaller hand:
# unitree_go::msg::dds_::MotorCmds_ with kMotorNum = 4, blended into
# ai_sport by weight. weight = cmds[0].mode / 100.
HAND_SDK_NUM_MOTORS = 4
HAND_SDK_ENCODED_NAME = 'unitree_go_msg_dds__MotorCmds_'

# =====================================================================
# TOPICS  (exact strings from the official examples)
# =====================================================================
TOPIC_LOW_CMD = 'rt/lowcmd'            # full-body 29-motor command
TOPIC_LOW_STATE = 'rt/lowstate'        # full-body state
TOPIC_ARM_SDK = 'rt/arm_sdk'           # arm/waist-only command (13 or 17)
TOPIC_IMU_TORSO = 'rt/secondary_imu'   # G1 torso IMU (IMUState_)
TOPIC_HAND_SDK = 'rt/hand_sdk'         # hand blend (MotorCmds_, 4 motors)
TOPIC_SPORT_MODE_STATE = 'rt/sportmodestate'

# =====================================================================
# DDS DOMAINS
# =====================================================================
DDS_DOMAIN_SIM = 1     # loopback sim; keeps sim traffic off the real robot
DDS_DOMAIN_REAL = 0    # real robot
DDS_INTERFACE_SIM = 'lo'

# =====================================================================
# GAINS - official values from g1_ankle_swing_example.cpp
#   Kp: 60,60,60,100,40,40 | 60,60,60,100,40,40 | 60,40,40
#       | 40 x7 | 40 x7
#   Kd: 1,1,1,2,1,1       | 1,1,1,2,1,1       | 1,1,1
#       | 1 x7 | 1 x7
# =====================================================================
OFFICIAL_KP = np.array([
    60.0, 60.0, 60.0, 100.0, 40.0, 40.0,
    60.0, 60.0, 60.0, 100.0, 40.0, 40.0,
    60.0, 40.0, 40.0,
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
    40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0,
], dtype=np.float32)

OFFICIAL_KD = np.array([
    1.0, 1.0, 1.0, 2.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 2.0, 1.0, 1.0,
    1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
], dtype=np.float32)

# Motor gearbox classes from g1_dual_arm_example.cpp
MOTOR_GEARBOX_S = 0
MOTOR_GEARBOX_M = 1
MOTOR_GEARBOX_L = 2

MOTOR_TYPES = np.array([
    MOTOR_GEARBOX_M, MOTOR_GEARBOX_M, MOTOR_GEARBOX_M, MOTOR_GEARBOX_L,
    MOTOR_GEARBOX_S, MOTOR_GEARBOX_S,
    MOTOR_GEARBOX_M, MOTOR_GEARBOX_M, MOTOR_GEARBOX_M, MOTOR_GEARBOX_L,
    MOTOR_GEARBOX_S, MOTOR_GEARBOX_S,
    MOTOR_GEARBOX_M, MOTOR_GEARBOX_S, MOTOR_GEARBOX_S,
] + [MOTOR_GEARBOX_S] * 14, dtype=np.int32)

KP_BY_GEARBOX = {MOTOR_GEARBOX_S: 40.0, MOTOR_GEARBOX_M: 40.0, MOTOR_GEARBOX_L: 100.0}
KD_BY_GEARBOX = {MOTOR_GEARBOX_S: 1.0, MOTOR_GEARBOX_M: 1.0, MOTOR_GEARBOX_L: 1.0}


def gains_from_gearbox() -> Tuple[np.ndarray, np.ndarray]:
    """Per-motor kp/kd derived from gearbox class (dual_arm example)."""
    kp = np.array([KP_BY_GEARBOX[int(t)] for t in MOTOR_TYPES], dtype=np.float32)
    kd = np.array([KD_BY_GEARBOX[int(t)] for t in MOTOR_TYPES], dtype=np.float32)
    return kp, kd


# Gains used by the high-level arm_sdk demos
ARM_SDK_KP = 60.0
ARM_SDK_KD = 1.5
ARM_SDK_MAX_JOINT_VELOCITY = 0.5  # rad/s, official
ARM_SDK_CONTROL_DT = 0.02          # 50 Hz, official
ARM_SDK_WEIGHT_RATE = 0.2          # weight ramp per second, official

# =====================================================================
# JOINT GROUPS
# =====================================================================
LEG_JOINTS: List[int] = list(range(0, 12))
WAIST_JOINTS: List[int] = [WAIST_YAW, WAIST_ROLL, WAIST_PITCH]
LEFT_ARM_JOINTS: List[int] = list(range(15, 22))
RIGHT_ARM_JOINTS: List[int] = list(range(22, 29))
UPPER_BODY_JOINTS: List[int] = WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

# arm_sdk publish sets (must be written on TOPIC_ARM_SDK)
# 17-DOF variant: both arms + waist   (g1_arm7_sdk_dds_example.cpp)
ARM_SDK_JOINTS_17: List[int] = UPPER_BODY_JOINTS
# 5-DOF-per-arm variant. NOTE the official C++ enum reuses slots 18/19 and
# 25/26 for elbow pitch/roll, which OVERLAP wrist_roll in the 29-DOF map.
# Only safe on a 23-DOF waist-locked arm. Kept for reference, not published.
ARM_SDK_JOINTS_13: List[int] = [
    15, 16, 17, 18, 19,
    22, 23, 24, 25, 26,
    WAIST_YAW, WAIST_ROLL, WAIST_PITCH,
]

# =====================================================================
# TORQUE LIMITS (Nm) - approximate, from gearbox class
# =====================================================================
TORQUE_LIMITS = np.array([
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
    88.0, 50.0, 50.0,
    25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0,
    25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0,
], dtype=np.float32)

# =====================================================================
# JOINT LIMITS (rad) - from the MuJoCo MJCF, not hand-typed.
# g1_robot.py reads these from the model; these are the fallback.
# =====================================================================
JOINT_LIMITS_MIN = np.array([
    -2.864, -0.428, -2.864, -0.428, -1.303, -0.612,
    -2.864, -0.428, -2.864, -0.428, -1.303, -0.612,
    -3.054, -1.413, -1.413,
    -3.142, -2.251, -2.251, -1.698, -3.072, -1.833, -3.072,
    -3.142, -2.251, -2.251, -1.698, -3.072, -1.833, -3.072,
], dtype=np.float32)

JOINT_LIMITS_MAX = np.array([
    2.864, 2.251, 2.864, 2.705, 0.613, 0.612,
    2.864, 2.251, 2.864, 2.705, 0.613, 0.612,
    3.054, 1.413, 1.413,
    1.298, 2.251, 2.251, 1.658, 3.072, 1.309, 3.072,
    1.298, 2.251, 2.251, 1.658, 3.072, 1.309, 3.072,
], dtype=np.float32)

# =====================================================================
# POSES
# =====================================================================
ZERO_POSE_Q = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)

NOMINAL_STAND_Q = np.array([
    -0.20, 0.0, 0.0, 0.40, -0.20, 0.0,
    -0.20, 0.0, 0.0, 0.40, -0.20, 0.0,
    0.0, 0.0, 0.0,
    0.30, 0.25, 0.0, 0.80, 0.0, 0.0, 0.0,
    0.30, -0.25, 0.0, 0.80, 0.0, 0.0, 0.0,
], dtype=np.float32)

# Arm rest pose used by g1_arm7_sdk_dds_example.cpp target_pos
ARM_REST_17 = np.zeros(17, dtype=np.float32)
ARM_REST_17[0:7] = [0.0, np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0, 0.0]
ARM_REST_17[7:14] = [0.0, -np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0, 0.0]

# =====================================================================
# CONTROL / SIM
# =====================================================================
LOWCMD_CONTROL_DT = 0.002   # 500 Hz, official for rt/lowcmd
LOWCMD_CONTROL_FREQ = 500
POLICY_FREQ = 50
POLICY_DT = 1.0 / POLICY_FREQ
PHYSICS_DT = 0.002
DECIMATION = LOWCMD_CONTROL_FREQ // POLICY_FREQ  # 10

MAX_JOINT_VELOCITY_FULL = 2.0  # rad/s budget for full-body path
JOINT_LIMIT_MARGIN = 0.03     # rad

# Ankle swing amplitudes used by g1_ankle_swing_example.cpp
ANKLE_SWING_P_MAX = float(np.pi * 30.0 / 180.0)
ANKLE_SWING_R_MAX = float(np.pi * 10.0 / 180.0)

# =====================================================================
# MODES
# =====================================================================
MODE_PR = 0  # series control for pitch/roll joints
MODE_AB = 1  # parallel control for A/B joints

MOTOR_MODE_DISABLE = 0
MOTOR_MODE_ENABLE = 1

# =====================================================================
# SAFETY
# =====================================================================
FALL_HEIGHT_THRESHOLD = 0.45
SELF_COLLISION_DIST = -0.002
ELASTIC_BAND_HEIGHT = 0.793

# =====================================================================
# CRC32  (dwPolynomial 0x04c11db7, init 0xFFFFFFFF, no final xor)
# ====================================================================
CRC32_POLYNOMIAL = 0x04C11DB7


def crc32_core(words: np.ndarray) -> int:
    """
    Bitwise CRC32 over a uint32 array. Mirrors the C++ Crc32Core used by the
    SDK. Provided for reference/tests; on-hardware always use
    unitree_sdk2py.utils.crc.CRC so packing matches the generated IDL struct.
    """
    crc = 0xFFFFFFFF
    for word in np.asarray(words, dtype=np.uint32).ravel():
        xbit = 1 << 31
        data = int(word)
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ CRC32_POLYNOMIAL) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if data & xbit:
                crc ^= CRC32_POLYNOMIAL
            xbit >>= 1
    return crc & 0xFFFFFFFF


# =====================================================================
# MUJOCO JOINT NAMES
# =====================================================================
JOINT_NAMES_29DOF: List[str] = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint', 'left_elbow_joint',
    'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint', 'right_elbow_joint',
    'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
]

# =====================================================================
# RPC SERVICE NAMES (motion_switcher)
# =====================================================================
SPORT_SERVICE_FORM = '0'
SPORT_SERVICE_NAMES = {
    'normal': 'sport_mode',
    'ai': 'ai_sport',
    'advanced': 'advanced_sport',
}

# =====================================================================
# HIGH-LEVEL ACTION MAPS
# =====================================================================
ARM_ACTIONS = {
    'release arm': 99, 'two-hand kiss': 11, 'left kiss': 12, 'right kiss': 13,
    'hands up': 15, 'clap': 17, 'high five': 18, 'hug': 19,
    'right hand wave': 20, 'right hand wave2': 21, 'shake hand': 22,
    'right hand salute': 23, 'left hand wave': 24, 'left hand wave2': 25,
    'left hand salute': 26, 'right hand raise': 27, 'left hand raise': 28,
    'both hands raise': 29, 'cross arms': 30,
}

LOCO_ACTIONS = {
    'damp': 0, 'Squat2StandUp': 1, 'StandUp2Squat': 2, 'move forward': 3,
    'move lateral': 4, 'move rotate': 5, 'low stand': 6, 'high stand': 7,
    'zero torque': 8, 'wave hand1': 9, 'wave hand2': 10, 'shake hand': 11,
    'Lie2StandUp': 12,
}


@dataclass
class G1Config:
    num_motors: int = NUM_BODY_MOTORS
    use_hands: bool = False
    fix_base: bool = True
    control_path: str = 'full'          # 'full' -> rt/lowcmd, 'arm' -> rt/arm_sdk
    dds_domain: int = DDS_DOMAIN_SIM
    dds_interface: str = DDS_INTERFACE_SIM
    policy_freq: int = POLICY_FREQ
    enable_elastic_band: bool = True
    elastic_band_height: float = ELASTIC_BAND_HEIGHT

    def __post_init__(self) -> None:
        if self.control_path not in ('full', 'arm'):
            raise ValueError("control_path must be 'full' or 'arm'")
        if self.num_motors != NUM_BODY_MOTORS:
            raise ValueError('G1 body is exactly 29 motors')

