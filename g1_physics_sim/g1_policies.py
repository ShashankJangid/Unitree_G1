from __future__ import annotations
import time
import numpy as np
from typing import Dict, Any, Optional
from g1_policy import Policy
from g1_state import G1State, G1Action, G1Observation
from g1_constants import (
    NUM_BODY_MOTORS, NOMINAL_STAND_Q, ZERO_POSE_Q,
    DEFAULT_KP, DEFAULT_KD, TORQUE_LIMITS,
    L_LEG_HIP_PITCH, L_LEG_HIP_ROLL, L_LEG_HIP_YAW, L_LEG_KNEE, L_LEG_ANKLE_PITCH, L_LEG_ANKLE_ROLL,
    R_LEG_HIP_PITCH, R_LEG_HIP_ROLL, R_LEG_HIP_YAW, R_LEG_KNEE, R_LEG_ANKLE_PITCH, R_LEG_ANKLE_ROLL,
    WAIST_YAW, WAIST_ROLL, WAIST_PITCH,
    L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW, L_ELBOW, L_WRIST_ROLL, L_WRIST_PITCH, L_WRIST_YAW,
    R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW, R_ELBOW, R_WRIST_ROLL, R_WRIST_PITCH, R_WRIST_YAW,
    LEFT_ARM_INDICES, RIGHT_ARM_INDICES, LEG_JOINT_INDICES, WAIST_INDICES
)


class AnkleSwingPolicy(Policy):
    """
    Reproduces Unitree's official ankle swing demo.
    Cycles through: zero pose -> PR mode swing -> AB mode swing with wrist roll.
    """

    def __init__(self, duration: float = 3.0, right_arm: bool = False):
        self.duration = duration
        self.right_arm = right_arm
        self.phase = 0
        self.phase_time = 0.0
        self.total_time = 0.0
        self.kp = 60.0
        self.kd = 1.5

        self.arm_joints = LEFT_ARM_INDICES if not right_arm else RIGHT_ARM_INDICES
        self.waist_joints = WAIST_INDICES

        self.target_pos = [
            0.0, np.pi/2, 0.0, np.pi/2, 0.0,
            0.0, -np.pi/2, 0.0, np.pi/2, 0.0,
            0.0, 0.0, 0.0
        ]

    def reset(self, state: G1State) -> None:
        self.phase = 0
        self.phase_time = 0.0
        self.total_time = 0.0

    def step(self, state: G1State) -> G1Action:
        self.total_time += 1.0 / 50.0
        self.phase_time += 1.0 / 50.0

        action = G1Action.zero(num_body=NUM_BODY_MOTORS)
        action.enable_arm_sdk = True

        if self.total_time < self.duration:

            ratio = np.clip(self.total_time / self.duration, 0.0, 1.0)
            action.q_des[:] = (1.0 - ratio) * state.q + ratio * ZERO_POSE_Q
            action.kp[:] = self.kp
            action.kd[:] = self.kd

        elif self.total_time < self.duration * 3:

            ratio = np.clip((self.total_time - self.duration) / (self.duration * 2), 0.0, 1.0)
            action.q_des[:] = ZERO_POSE_Q.copy()
            for i, joint in enumerate(self.arm_joints + self.waist_joints):
                if i < len(self.target_pos):
                    action.q_des[joint] = ratio * self.target_pos[i]
            action.kp[:] = self.kp
            action.kd[:] = self.kd

        elif self.total_time < self.duration * 6:

            t = self.total_time - self.duration * 3
            action.q_des[:] = ZERO_POSE_Q.copy()
            for i, joint in enumerate(self.arm_joints + self.waist_joints):
                if i < len(self.target_pos):
                    action.q_des[joint] = self.target_pos[i]


            swing_amp = 0.3
            action.q_des[L_LEG_ANKLE_PITCH] = swing_amp * np.sin(2 * np.pi * 0.5 * t)
            action.q_des[L_LEG_ANKLE_ROLL] = swing_amp * np.sin(2 * np.pi * 0.5 * t + np.pi/2)
            action.q_des[R_LEG_ANKLE_PITCH] = swing_amp * np.sin(2 * np.pi * 0.5 * t)
            action.q_des[R_LEG_ANKLE_ROLL] = -swing_amp * np.sin(2 * np.pi * 0.5 * t + np.pi/2)

            action.kp[:] = self.kp
            action.kd[:] = self.kd

        elif self.total_time < self.duration * 9:

            t = self.total_time - self.duration * 6
            action.q_des[:] = ZERO_POSE_Q.copy()
            for i, joint in enumerate(self.arm_joints + self.waist_joints):
                if i < len(self.target_pos):
                    action.q_des[joint] = self.target_pos[i]


            swing_amp = 0.3
            action.q_des[L_LEG_HIP_ROLL] = swing_amp * np.sin(2 * np.pi * 0.5 * t)
            action.q_des[L_LEG_HIP_YAW] = swing_amp * np.sin(2 * np.pi * 0.5 * t + np.pi/2)
            action.q_des[R_LEG_HIP_ROLL] = -swing_amp * np.sin(2 * np.pi * 0.5 * t)
            action.q_des[R_LEG_HIP_YAW] = -swing_amp * np.sin(2 * np.pi * 0.5 * t + np.pi/2)


            wrist_joint = R_WRIST_ROLL if self.right_arm else L_WRIST_ROLL
            action.q_des[wrist_joint] = 0.5 * np.sin(2 * np.pi * 1.0 * t)

            action.kp[:] = self.kp
            action.kd[:] = self.kd

        else:

            ratio = np.clip((self.total_time - self.duration * 9) / self.duration, 0.0, 1.0)
            action.enable_arm_sdk = (1.0 - ratio) > 0.5
            action.q_des[:] = state.q
            action.kp[:] = self.kp
            action.kd[:] = self.kd

        return action

    def get_observation(self, state: G1State) -> G1Observation:
        return G1Observation(
            timestamp=state.timestamp,
            joint_pos=state.q,
            joint_vel=state.dq,
            base_linvel=state.base_linvel,
            base_angvel=state.base_angvel,
            base_quat=state.base_quat,
            imu_angvel=state.imu_angvel,
            imu_linacc=state.imu_linacc,
            left_foot_contact=state.left_foot_contact,
            right_foot_contact=state.right_foot_contact,
        )

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return 29 * 2 + 3 + 3 + 4 + 3 + 3 + 2

    def get_info(self) -> Dict[str, Any]:
        return {
            'phase': self.phase,
            'phase_time': self.phase_time,
            'total_time': self.total_time,
        }


class StandPolicy(Policy):
    """Simple stand-still policy with configurable posture."""

    def __init__(
        self,
        target_q: Optional[np.ndarray] = None,
        kp: float = 60.0,
        kd: float = 1.5,
        enable_arm_sdk: bool = True
    ):
        self.target_q = target_q if target_q is not None else NOMINAL_STAND_Q.copy()
        self.kp_val = kp
        self.kd_val = kd
        self.enable_arm_sdk = enable_arm_sdk

    def reset(self, state: G1State) -> None:
        pass

    def step(self, state: G1State) -> G1Action:
        action = G1Action.zero(num_body=NUM_BODY_MOTORS)
        action.q_des[:] = self.target_q[:NUM_BODY_MOTORS]
        action.kp[:] = self.kp_val
        action.kd[:] = self.kd_val
        action.enable_arm_sdk = self.enable_arm_sdk
        return action

    def get_observation(self, state: G1State) -> G1Observation:
        return G1Observation(
            timestamp=state.timestamp,
            joint_pos=state.q,
            joint_vel=state.dq,
            base_linvel=state.base_linvel,
            base_angvel=state.base_angvel,
            base_quat=state.base_quat,
            imu_angvel=state.imu_angvel,
            imu_linacc=state.imu_linacc,
            left_foot_contact=state.left_foot_contact,
            right_foot_contact=state.right_foot_contact,
        )

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return 29 * 2 + 3 + 3 + 4 + 3 + 3 + 2


class ArmTrackingPolicy(Policy):
    """Cartesian-space arm tracking with IK."""

    def __init__(
        self,
        target_pos: np.ndarray,
        right_arm: bool = False,
        kp: float = 60.0,
        kd: float = 1.5,
        cart_gain: float = 0.1,
        nullspace_gain: float = 0.2
    ):
        self.target_pos = target_pos
        self.right_arm = right_arm
        self.kp_val = kp
        self.kd_val = kd
        self.cart_gain = cart_gain
        self.nullspace_gain = nullspace_gain

        self.arm_indices = RIGHT_ARM_INDICES if right_arm else LEFT_ARM_INDICES
        self.idle_indices = LEFT_ARM_INDICES if right_arm else RIGHT_ARM_INDICES

    def reset(self, state: G1State) -> None:
        pass

    def step(self, state: G1State) -> G1Action:

        action = G1Action.zero(num_body=NUM_BODY_MOTORS)
        action.q_des[:] = state.q.copy()


        arm_target = np.array([0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0])
        alpha = 0.1
        action.q_des[self.arm_indices] = (1 - alpha) * state.q[self.arm_indices] + alpha * arm_target
        action.q_des[self.idle_indices] = NOMINAL_STAND_Q[self.idle_indices]

        action.kp[:] = self.kp_val
        action.kd[:] = self.kd_val
        action.enable_arm_sdk = True
        return action

    def get_observation(self, state: G1State) -> G1Observation:
        return G1Observation(
            timestamp=state.timestamp,
            joint_pos=state.q,
            joint_vel=state.dq,
            base_linvel=state.base_linvel,
            base_angvel=state.base_angvel,
            base_quat=state.base_quat,
            imu_angvel=state.imu_angvel,
            imu_linacc=state.imu_linacc,
            left_foot_contact=state.left_foot_contact,
            right_foot_contact=state.right_foot_contact,
        )

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return 29 * 2 + 3 + 3 + 4 + 3 + 3 + 2


class JointPositionPolicy(Policy):
    """Simple joint position control policy for testing."""

    def __init__(
        self,
        target_q: np.ndarray,
        kp: float = 60.0,
        kd: float = 1.5,
        ramp_time: float = 2.0
    ):
        self.target_q = target_q
        self.kp_val = kp
        self.kd_val = kd
        self.ramp_time = ramp_time
        self.start_time = 0.0

    def reset(self, state: G1State) -> None:
        self.start_time = state.timestamp

    def step(self, state: G1State) -> G1Action:
        elapsed = state.timestamp - self.start_time
        ratio = np.clip(elapsed / self.ramp_time, 0.0, 1.0)

        action = G1Action.zero(num_body=NUM_BODY_MOTORS)
        action.q_des[:] = (1.0 - ratio) * state.q + ratio * self.target_q[:NUM_BODY_MOTORS]
        action.kp[:] = self.kp_val
        action.kd[:] = self.kd_val
        action.enable_arm_sdk = True
        return action

    def get_observation(self, state: G1State) -> G1Observation:
        return G1Observation(
            timestamp=state.timestamp,
            joint_pos=state.q,
            joint_vel=state.dq,
            base_linvel=state.base_linvel,
            base_angvel=state.base_angvel,
            base_quat=state.base_quat,
            imu_angvel=state.imu_angvel,
            imu_linacc=state.imu_linacc,
            left_foot_contact=state.left_foot_contact,
            right_foot_contact=state.right_foot_contact,
        )

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return 29 * 2 + 3 + 3 + 4 + 3 + 3 + 2


class LocomotionPolicy(Policy):
    """Base class for locomotion policies (to be implemented with RL)."""

    def __init__(self, policy_path: Optional[str] = None):
        self.policy_path = policy_path
        self._onnx_session = None
        if policy_path:
            self._load_policy()

    def _load_policy(self):
        try:
            import onnxruntime as ort
            self._onnx_session = ort.InferenceSession(self.policy_path)
        except Exception as e:
            print(f"[LocomotionPolicy] Failed to load policy: {e}")

    def reset(self, state: G1State) -> None:
        pass

    def step(self, state: G1State) -> G1Action:
        action = G1Action.zero(num_body=NUM_BODY_MOTORS)
        action.q_des[:] = state.q.copy()
        action.kp[:] = DEFAULT_KP[:NUM_BODY_MOTORS]
        action.kd[:] = DEFAULT_KD[:NUM_BODY_MOTORS]
        action.enable_arm_sdk = False

        if self._onnx_session is not None:
            obs = self._get_policy_obs(state)
            try:
                action_output = self._onnx_session.run(None, {'obs': obs.astype(np.float32)})[0]
                action.q_des[LEG_JOINT_INDICES] = action_output[:12]
            except Exception as e:
                print(f"[LocomotionPolicy] Inference error: {e}")

        return action

    def _get_policy_obs(self, state: G1State) -> np.ndarray:

        obs_parts = [
            state.q[LEG_JOINT_INDICES],
            state.dq[LEG_JOINT_INDICES],
            state.base_linvel,
            state.base_angvel,
            state.imu_angvel,
            state.imu_linacc,
            np.array([float(state.left_foot_contact), float(state.right_foot_contact)], dtype=np.float32),
        ]
        return np.concatenate(obs_parts)

    def get_observation(self, state: G1State) -> G1Observation:
        return G1Observation(
            timestamp=state.timestamp,
            joint_pos=state.q,
            joint_vel=state.dq,
            base_linvel=state.base_linvel,
            base_angvel=state.base_angvel,
            base_quat=state.base_quat,
            imu_angvel=state.imu_angvel,
            imu_linacc=state.imu_linacc,
            left_foot_contact=state.left_foot_contact,
            right_foot_contact=state.right_foot_contact,
        )

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return 12 * 2 + 3 + 3 + 3 + 3 + 2


class WavePolicy(Policy):
    """Friendly wave gesture policy."""

    def __init__(
        self,
        right_arm: bool = False,
        duration: float = 4.0,
        kp: float = 40.0,
        kd: float = 1.0
    ):
        self.right_arm = right_arm
        self.duration = duration
        self.kp_val = kp
        self.kd_val = kd
        self.start_time = 0.0

        self.arm_indices = RIGHT_ARM_INDICES if right_arm else LEFT_ARM_INDICES
        self.idle_indices = LEFT_ARM_INDICES if right_arm else RIGHT_ARM_INDICES

    def reset(self, state: G1State) -> None:
        self.start_time = state.timestamp

    def step(self, state: G1State) -> G1Action:
        t = state.timestamp - self.start_time
        phase = t / self.duration

        action = G1Action.zero(num_body=NUM_BODY_MOTORS)
        action.q_des[:] = NOMINAL_STAND_Q.copy()
        action.q_des[self.idle_indices] = NOMINAL_STAND_Q[self.idle_indices]


        pitch_idx = self.arm_indices[0]
        roll_idx = self.arm_indices[1]
        elbow_idx = self.arm_indices[3]
        wrist_roll_idx = self.arm_indices[4]
        yaw_idx = self.arm_indices[2]

        side = -1.0 if self.right_arm else 1.0

        action.q_des[pitch_idx] = -0.6
        action.q_des[roll_idx] = side * 0.7
        action.q_des[elbow_idx] = 1.2
        action.q_des[wrist_roll_idx] = 0.45 * np.sin(2.0 * np.pi * 2.5 * t)
        action.q_des[yaw_idx] = side * (0.2 + 0.15 * np.cos(2.0 * np.pi * 2.5 * t))

        action.kp[:] = self.kp_val
        action.kd[:] = self.kd_val
        action.enable_arm_sdk = True

        return action

    def get_observation(self, state: G1State) -> G1Observation:
        return G1Observation(
            timestamp=state.timestamp,
            joint_pos=state.q,
            joint_vel=state.dq,
            base_linvel=state.base_linvel,
            base_angvel=state.base_angvel,
            base_quat=state.base_quat,
            imu_angvel=state.imu_angvel,
            imu_linacc=state.imu_linacc,
            left_foot_contact=state.left_foot_contact,
            right_foot_contact=state.right_foot_contact,
        )

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return 29 * 2 + 3 + 3 + 4 + 3 + 3 + 2
