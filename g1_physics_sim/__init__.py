"""
Unitree G1 - physics simulation and real-robot control.

The body is exactly 29 motors. Dexterous hands are a separate subsystem on a
separate DDS topic with a separate message type and are NOT appended to the body
index space.

Quick start
-----------
Simulation:
    python run_sim.py --policy arm_swing

Real robot (developer mode, gantry, E-stop in hand):
    python run_real.py --policy stand --iface eth0 --path arm

Only what is verified against the official SDK is exported here. Anything
hand-tuned or invented during prototyping was removed rather than aliased, so an
ImportError names a real problem instead of silently substituting a wrong value.
"""
from .g1_constants import (
    # identity
    NUM_BODY_MOTORS, LOWCMD_SLOTS,
    # joint indices (official G1JointIndex)
    L_HIP_PITCH, L_HIP_ROLL, L_HIP_YAW, L_KNEE, L_ANKLE_PITCH, L_ANKLE_ROLL,
    R_HIP_PITCH, R_HIP_ROLL, R_HIP_YAW, R_KNEE, R_ANKLE_PITCH, R_ANKLE_ROLL,
    WAIST_YAW, WAIST_ROLL, WAIST_PITCH,
    L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW, L_ELBOW,
    L_WRIST_ROLL, L_WRIST_PITCH, L_WRIST_YAW,
    R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW, R_ELBOW,
    R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW,
    # PR/AB aliases
    L_ANKLE_A, L_ANKLE_B, R_ANKLE_A, R_ANKLE_B, WAIST_A, WAIST_B,
    # arm_sdk
    K_NOT_USED_JOINT, ARM_SDK_ENABLE, ARM_SDK_JOINTS_17, ARM_SDK_JOINTS_13,
    # hands (separate topic)
    DEX3_JOINTS_PER_HAND, DEX3_NUM_MOTORS, DEX3_JOINT_NAMES, HAND_SDK_NUM_MOTORS,
    # groups
    LEG_JOINTS, WAIST_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, UPPER_BODY_JOINTS,
    # gains / limits
    OFFICIAL_KP, OFFICIAL_KD, MOTOR_TYPES, gains_from_gearbox,
    TORQUE_LIMITS, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX,
    ARM_SDK_KP, ARM_SDK_KD, ARM_SDK_MAX_JOINT_VELOCITY,
    ARM_SDK_CONTROL_DT, ARM_SDK_WEIGHT_RATE, LOWCMD_CONTROL_DT,
    MAX_JOINT_VELOCITY_FULL, JOINT_LIMIT_MARGIN, POLICY_FREQ, POLICY_DT,
    ANKLE_SWING_P_MAX, ANKLE_SWING_R_MAX,
    # topics / domains
    TOPIC_LOW_CMD, TOPIC_LOW_STATE, TOPIC_ARM_SDK, TOPIC_IMU_TORSO, TOPIC_HAND_SDK,
    DDS_DOMAIN_SIM, DDS_DOMAIN_REAL, DDS_INTERFACE_SIM,
    # modes
    MODE_PR, MODE_AB, MOTOR_MODE_DISABLE, MOTOR_MODE_ENABLE,
    # poses
    NOMINAL_STAND_Q, ZERO_POSE_Q, ARM_REST_17, JOINT_NAMES_29DOF,
    # safety
    FALL_HEIGHT_THRESHOLD, SELF_COLLISION_DIST, ELASTIC_BAND_HEIGHT,
    # misc
    crc32_core, G1Config, ARM_ACTIONS, LOCO_ACTIONS,
)
from .g1_state import G1Action, G1Observation, G1State, quat_to_rpy, rpy_to_quat
from .g1_hardware_bridge import (
    G1Bridge, G1RealHardwareBridge, G1SimBridge, SDK_AVAILABLE, create_g1_bridge,
)
from .g1_policy import Policy, UnitreeG1Robot, run_policy
from .g1_policies import (
    AnkleSwingPolicy, ArmSwingPolicy, JointPositionPolicy,
    LocomotionPolicy, StandPolicy, WavePolicy,
)
from .g1_trajectory_manager import G1MovementScripts, TrajectoryPlayer, TrajectoryRecorder
from .g1_safety import (
    G1FullSafetyManager, SafetyState, SafetyStatus, SCurveLimiter,
    SelfCollisionGuard, ThermalWatchdog,
)

__version__ = '2.0.0'

__all__ = [
    'G1Robot',
    'Policy', 'UnitreeG1Robot', 'run_policy',
    'G1Bridge', 'G1SimBridge', 'G1RealHardwareBridge', 'create_g1_bridge', 'SDK_AVAILABLE',
    'G1State', 'G1Action', 'G1Observation', 'rpy_to_quat', 'quat_to_rpy',
    'StandPolicy', 'ArmSwingPolicy', 'WavePolicy', 'AnkleSwingPolicy',
    'JointPositionPolicy', 'LocomotionPolicy',
    'TrajectoryRecorder', 'TrajectoryPlayer', 'G1MovementScripts',
    'G1FullSafetyManager', 'SafetyState', 'SafetyStatus',
    'SCurveLimiter', 'SelfCollisionGuard', 'ThermalWatchdog',
    'G1Config', 'crc32_core', 'gains_from_gearbox',
]


def __getattr__(name: str):
    """Lazy-import G1Robot so `import g1_physics_sim` works without mujoco."""
    if name == 'G1Robot':
        from .g1_robot import G1Robot as _G1Robot
        return _G1Robot
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')