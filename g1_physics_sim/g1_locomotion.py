from __future__ import annotations
import pathlib
from typing import Optional, Tuple
import numpy as np
import mujoco

try:
    from .g1_constants import (
        NUM_BODY_MOTORS, NOMINAL_STAND_Q,
        L_HIP_PITCH, L_HIP_ROLL, L_HIP_YAW, L_KNEE, L_ANKLE_PITCH, L_ANKLE_ROLL,
        R_HIP_PITCH, R_HIP_ROLL, R_HIP_YAW, R_KNEE, R_ANKLE_PITCH, R_ANKLE_ROLL,
        WAIST_YAW, WAIST_ROLL, WAIST_PITCH,
        L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW, L_ELBOW, L_WRIST_ROLL, L_WRIST_PITCH, L_WRIST_YAW,
        R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW, R_ELBOW, R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW,
    )
except ImportError:
    from g1_constants import (
        NUM_BODY_MOTORS, NOMINAL_STAND_Q,
        L_HIP_PITCH, L_HIP_ROLL, L_HIP_YAW, L_KNEE, L_ANKLE_PITCH, L_ANKLE_ROLL,
        R_HIP_PITCH, R_HIP_ROLL, R_HIP_YAW, R_KNEE, R_ANKLE_PITCH, R_ANKLE_ROLL,
        WAIST_YAW, WAIST_ROLL, WAIST_PITCH,
        L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW, L_ELBOW, L_WRIST_ROLL, L_WRIST_PITCH, L_WRIST_YAW,
        R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW, R_ELBOW, R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW,
    )


class G1LocomotionEngine:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.is_active = False
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_z = 0.0
        self.cur_vx = 0.0
        self.cur_vy = 0.0
        self.cur_vyaw = 0.0
        self.phase = 0.0
        self.gait_freq = 1.3
        self.step_count = 0
        self.default_joint_pos = NOMINAL_STAND_Q.copy()
        self.target_joint_pos = NOMINAL_STAND_Q.copy()

    def reset(self):
        self.step_count = 0
        self.phase = 0.0
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_z = 0.0
        self.cur_vx = 0.0
        self.cur_vy = 0.0
        self.cur_vyaw = 0.0
        self.target_joint_pos = NOMINAL_STAND_Q.copy()

    def set_velocity_command(self, vx: float, vy: float, vyaw: float):
        self.lin_vel_x = float(np.clip(vx, -0.6, 1.2))
        self.lin_vel_y = float(np.clip(vy, -0.5, 0.5))
        self.ang_vel_z = float(np.clip(vyaw, -1.0, 1.0))

    def step(self, dt: float = 0.02) -> np.ndarray:
        if not self.is_active:
            return NOMINAL_STAND_Q.copy()

        alpha = 0.15
        self.cur_vx += alpha * (self.lin_vel_x - self.cur_vx)
        self.cur_vy += alpha * (self.lin_vel_y - self.cur_vy)
        self.cur_vyaw += alpha * (self.ang_vel_z - self.cur_vyaw)

        speed = np.sqrt(self.cur_vx**2 + self.cur_vy**2 + (0.5 * self.cur_vyaw)**2)
        if speed > 0.02:
            self.phase = (self.phase + 2.0 * np.pi * self.gait_freq * dt) % (2.0 * np.pi)
        else:
            self.phase = 0.0

        q = NOMINAL_STAND_Q.copy()
        if speed <= 0.01:
            self.target_joint_pos = q
            self.step_count += 1
            return q

        s = np.sin(self.phase)
        c = np.cos(self.phase)

        step_len = self.cur_vx * 0.40
        sway = 0.035 * s

        q[L_HIP_ROLL] = sway + self.cur_vy * 0.15
        q[R_HIP_ROLL] = -sway + self.cur_vy * 0.15
        q[L_ANKLE_ROLL] = -sway * 0.7
        q[R_ANKLE_ROLL] = sway * 0.7

        l_sw = max(0.0, s)
        r_sw = max(0.0, -s)

        q[L_HIP_PITCH] += -step_len * c - 0.22 * l_sw
        q[R_HIP_PITCH] += step_len * c - 0.22 * r_sw

        q[L_KNEE] += 0.45 * l_sw
        q[R_KNEE] += 0.45 * r_sw

        q[L_ANKLE_PITCH] += -0.22 * l_sw + 0.15 * c * step_len
        q[R_ANKLE_PITCH] += -0.22 * r_sw - 0.15 * c * step_len

        yaw_bias = 0.12 * self.cur_vyaw * s
        q[L_HIP_YAW] += yaw_bias + 0.2 * self.cur_vyaw
        q[R_HIP_YAW] += yaw_bias - 0.2 * self.cur_vyaw
        q[WAIST_YAW] = -0.1 * yaw_bias

        arm_swing = step_len * 0.7
        q[L_SHOULDER_PITCH] += arm_swing * c
        q[R_SHOULDER_PITCH] -= arm_swing * c

        self.target_joint_pos = q
        self.step_count += 1
        return q
