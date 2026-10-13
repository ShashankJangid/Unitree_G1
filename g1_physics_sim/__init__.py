"""
Unitree G1 Physics Simulation & Real Robot Control
===================================================

A unified framework for sim-to-real deployment on the Unitree G1 humanoid robot.

Features:
- MuJoCo simulation with accurate G1 dynamics
- Real robot control via Unitree SDK2 Python (DDS/CycloneDDS)
- Sim-to-real deployment with identical code paths
- Two-thread architecture: 500Hz control + policy rate
- Safety monitoring and emergency stop
- Trajectory recording and playback
- Multiple policy examples: stand, walk, wave, ankle swing, etc.

Quick Start (Simulation):
    python run_sim.py --policy ankle_swing

Quick Start (Real Robot):
    python run_real.py --policy stand --interface eth0

Architecture:
    g1_constants.py     - Joint indices, PD gains, torque limits, DDS topics
    g1_robot.py         - MuJoCo robot model with elastic band support
    g1_hardware_bridge.py - Abstract bridge + sim/hardware implementations
    g1_state.py         - State/Action/Observation dataclasses
    g1_policy.py        - Abstract Policy interface + UnitreeG1Robot runner
    g1_policies.py      - Example policies (stand, wave, ankle_swing, etc.)
    g1_trajectory_manager.py - Recording/playback utilities
    config.yaml         - Configuration file
    config_loader.py    - Configuration management

Sim-to-Real Workflow:
    1. Develop policy in simulation (DDS domain 1, loopback interface)
    2. Test thoroughly with safety checks
    3. Deploy to real robot (DDS domain 0, ethernet interface)
    4. Same policy code runs on both!
"""

from .g1_constants import *
from .g1_robot import G1Robot
from .g1_hardware_bridge import G1Bridge, G1SimBridge, G1RealHardwareBridge, create_g1_bridge
from .g1_state import G1State, G1Action, G1Observation
from .g1_policy import Policy, UnitreeG1Robot, run_policy
from .g1_policies import (
    StandPolicy, AnkleSwingPolicy, WavePolicy,
    JointPositionPolicy, LocomotionPolicy
)
from .g1_trajectory_manager import TrajectoryRecorder, TrajectoryPlayer, G1MovementScripts
from .config_loader import load_config, G1Config, create_local_config

__version__ = "1.0.0"
__author__ = "Unitree G1 Research"

__all__ = [

    "NUM_BODY_MOTORS", "NUM_MOTORS", "JOINT_NAMES_29DOF", "JOINT_NAMES_43DOF",
    "DEFAULT_KP", "DEFAULT_KD", "TORQUE_LIMITS", "NOMINAL_STAND_Q", "ZERO_POSE_Q",
    "L_LEG_HIP_PITCH", "L_LEG_HIP_ROLL", "L_LEG_HIP_YAW", "L_LEG_KNEE", "L_LEG_ANKLE_PITCH", "L_LEG_ANKLE_ROLL",
    "R_LEG_HIP_PITCH", "R_LEG_HIP_ROLL", "R_LEG_HIP_YAW", "R_LEG_KNEE", "R_LEG_ANKLE_PITCH", "R_LEG_ANKLE_ROLL",
    "WAIST_YAW", "WAIST_ROLL", "WAIST_PITCH",
    "L_SHOULDER_PITCH", "L_SHOULDER_ROLL", "L_SHOULDER_YAW", "L_ELBOW", "L_WRIST_ROLL", "L_WRIST_PITCH", "L_WRIST_YAW",
    "R_SHOULDER_PITCH", "R_SHOULDER_ROLL", "R_SHOULDER_YAW", "R_ELBOW", "R_WRIST_ROLL", "R_WRIST_PITCH", "R_WRIST_YAW",
    "ARM_SDK_ENABLE", "LEG_JOINT_INDICES", "LEFT_ARM_INDICES", "RIGHT_ARM_INDICES", "WAIST_INDICES",
    "DDS_DOMAIN_SIM", "DDS_DOMAIN_REAL", "TOPIC_LOW_CMD", "TOPIC_LOW_STATE",


    "G1Robot",
    "G1Bridge", "G1SimBridge", "G1RealHardwareBridge", "create_g1_bridge",
    "G1State", "G1Action", "G1Observation",
    "Policy", "UnitreeG1Robot", "run_policy",


    "StandPolicy", "AnkleSwingPolicy", "WavePolicy",
    "JointPositionPolicy", "LocomotionPolicy",


    "TrajectoryRecorder", "TrajectoryPlayer", "G1MovementScripts",


    "load_config", "G1Config", "create_local_config",
]
