from dataclasses import dataclass
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
NUM_MOTORS = 29
JOINT_NAMES_29DOF: List[str] = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
DEFAULT_KP = np.array([100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 100.0, 100.0, 100.0, 150.0, 40.0, 40.0, 100.0, 60.0, 60.0, 40.0, 40.0, 40.0, 40.0, 25.0, 25.0, 25.0, 40.0, 40.0, 40.0, 40.0, 25.0, 25.0, 25.0], dtype=np.float32)
DEFAULT_KD = np.array([3.0, 3.0, 3.0, 4.0, 1.5, 1.5, 3.0, 3.0, 3.0, 4.0, 1.5, 1.5, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8], dtype=np.float32)
TORQUE_LIMITS = np.array([88.0, 88.0, 88.0, 139.0, 50.0, 50.0, 88.0, 88.0, 88.0, 139.0, 50.0, 50.0, 88.0, 50.0, 50.0, 25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 25.0, 25.0, 25.0, 25.0, 25.0, 5.0, 5.0], dtype=np.float32)
NOMINAL_STAND_Q = np.zeros(NUM_MOTORS, dtype=np.float32)
NOMINAL_STAND_Q[L_LEG_HIP_PITCH] = -0.28
NOMINAL_STAND_Q[L_LEG_KNEE] = 0.6
NOMINAL_STAND_Q[L_LEG_ANKLE_PITCH] = -0.32
NOMINAL_STAND_Q[R_LEG_HIP_PITCH] = -0.28
NOMINAL_STAND_Q[R_LEG_KNEE] = 0.6
NOMINAL_STAND_Q[R_LEG_ANKLE_PITCH] = -0.32
NOMINAL_STAND_Q[L_SHOULDER_PITCH] = 0.35
NOMINAL_STAND_Q[L_SHOULDER_ROLL] = 0.18
NOMINAL_STAND_Q[L_ELBOW] = 0.5
NOMINAL_STAND_Q[R_SHOULDER_PITCH] = 0.35
NOMINAL_STAND_Q[R_SHOULDER_ROLL] = -0.18
NOMINAL_STAND_Q[R_ELBOW] = 0.5
