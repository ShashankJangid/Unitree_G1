"""Inference-only adapter for the published GRIT G1 whole-body policy.

The policy and reference motion are external, pretrained artifacts.  This
module intentionally contains no optimizer, replay buffer, or training path.
It only builds the published observation contract and returns joint targets.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np


GRIT_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

CONTEXT_OFFSETS = np.array([0, 2, 3, 5, 7, 8, 10, 12, 13], dtype=np.int64)
HISTORY_LENGTH = 10

GRIT_ACTION_SCALE = np.array([
    0.5475464629911068, 0.35066146637882434, 0.5475464629911068,
    0.35066146637882434, 0.43857731392336724, 0.43857731392336724,
    0.5475464629911068, 0.35066146637882434, 0.5475464629911068,
    0.35066146637882434, 0.43857731392336724, 0.43857731392336724,
    0.5475464629911068, 0.43857731392336724, 0.43857731392336724,
    0.43857731392336724, 0.43857731392336724, 0.43857731392336724,
    0.43857731392336724, 0.43857731392336724, 0.07450087032950714,
    0.07450087032950714, 0.43857731392336724, 0.43857731392336724,
    0.43857731392336724, 0.43857731392336724, 0.43857731392336724,
    0.07450087032950714, 0.07450087032950714,
], dtype=np.float32)

GRIT_DEFAULT_Q = np.array([
    -0.312, 0.2, 0.0, 0.669, -0.363, 0.0,
    -0.312, -0.2, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float32)

GRIT_KP = np.array([
    40.17923863450712, 99.09842777666111, 40.17923863450712,
    99.09842777666111, 28.50124619574858, 28.50124619574858,
    40.17923863450712, 99.09842777666111, 40.17923863450712,
    99.09842777666111, 28.50124619574858, 28.50124619574858,
    40.17923863450712, 28.50124619574858, 28.50124619574858,
    14.25062309787429, 14.25062309787429, 14.25062309787429,
    14.25062309787429, 14.25062309787429, 16.77832748089279,
    16.77832748089279, 14.25062309787429, 14.25062309787429,
    14.25062309787429, 14.25062309787429, 14.25062309787429,
    16.77832748089279, 16.77832748089279,
], dtype=np.float32)

GRIT_KD = np.array([
    2.557889775413375, 6.308801853496639, 2.557889775413375,
    6.308801853496639, 1.814445686584846, 1.814445686584846,
    2.557889775413375, 6.308801853496639, 2.557889775413375,
    6.308801853496639, 1.814445686584846, 1.814445686584846,
    2.557889775413375, 1.814445686584846, 1.814445686584846,
    0.907222843292423, 0.907222843292423, 0.907222843292423,
    0.907222843292423, 0.907222843292423, 1.06814150219,
    1.06814150219, 0.907222843292423, 0.907222843292423,
    0.907222843292423, 0.907222843292423, 0.907222843292423,
    1.06814150219, 1.06814150219,
], dtype=np.float32)


def _rotation_matrix_wxyz(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-9)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _rot6d(quat: np.ndarray) -> np.ndarray:
    return _rotation_matrix_wxyz(quat)[..., :, :2].reshape(-1, 6).astype(np.float32)


def _projected_gravity(quat: np.ndarray) -> np.ndarray:
    rotation = _rotation_matrix_wxyz(quat)
    return np.einsum("nij,j->ni", np.swapaxes(rotation, 1, 2), [0.0, 0.0, -1.0]).astype(np.float32)


class GritDancePolicy:
    """Run the published GRIT ONNX graph against a local MuJoCo state."""

    def __init__(self, policy_path: Path, motion_path: Path, joint_names: list[str]):
        if list(joint_names) != GRIT_JOINT_NAMES:
            raise ValueError("GRIT policy joint order does not match the MuJoCo controller order")
        if not policy_path.is_file():
            raise FileNotFoundError(f"GRIT policy not found: {policy_path}")
        if not motion_path.is_file():
            raise FileNotFoundError(f"GRIT reference motion not found: {motion_path}")

        import onnxruntime as ort

        with policy_path.with_suffix(".json").open("r", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        expected_hash = self.metadata.get("onnx_sha256")
        if expected_hash:
            actual_hash = hashlib.sha256(policy_path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"GRIT ONNX hash mismatch: {actual_hash} != {expected_hash}")
        self.session = ort.InferenceSession(
            str(policy_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = tuple(item.name for item in self.session.get_inputs())
        outputs = tuple(item.name for item in self.session.get_outputs())
        if inputs != ("reference_context", "proprio_history"):
            raise ValueError(f"Unsupported GRIT inputs {inputs}")
        if outputs != ("actions",):
            raise ValueError(f"Unsupported GRIT outputs {outputs}")

        data = np.load(motion_path, allow_pickle=False)
        required = {"fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"GRIT motion is missing fields: {missing}")
        self.fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"GRIT motion fps must be positive and finite, got {self.fps}")
        self.joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        self.joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_lin_vel = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
        body_ang_vel = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
        frame_count = self.joint_pos.shape[0]
        if self.joint_pos.shape != (frame_count, 29) or self.joint_vel.shape != (frame_count, 29):
            raise ValueError(f"GRIT joint arrays must be [T, 29], got {self.joint_pos.shape} and {self.joint_vel.shape}")
        for name, array in (("body_pos_w", body_pos), ("body_lin_vel_w", body_lin_vel), ("body_ang_vel_w", body_ang_vel)):
            if array.shape[0] != frame_count or array.ndim != 3 or array.shape[1] < 1 or array.shape[2] != 3:
                raise ValueError(f"Unexpected GRIT {name} shape: {array.shape}")
        if body_quat.shape[0] != frame_count or body_quat.ndim != 3 or body_quat.shape[1] < 1 or body_quat.shape[2] != 4:
            raise ValueError(f"Unexpected GRIT body_quat_w shape: {body_quat.shape}")
        if not np.all(np.isfinite(self.joint_pos)) or not np.all(np.isfinite(self.joint_vel)):
            raise ValueError("GRIT joint motion contains non-finite values")
        root_pos = body_pos[:, 0, :]
        root_quat = body_quat[:, 0, :]
        if not np.all(np.isfinite(root_pos)) or not np.all(np.isfinite(root_quat)):
            raise ValueError("GRIT root motion contains non-finite values")
        root_quat /= np.maximum(np.linalg.norm(root_quat, axis=1, keepdims=True), 1e-9)
        root_rotation = _rotation_matrix_wxyz(root_quat)
        root_lin_vel = np.asarray(data["body_lin_vel_w"][:, 0, :], dtype=np.float32)
        root_ang_vel = np.asarray(data["body_ang_vel_w"][:, 0, :], dtype=np.float32)
        root_lin_vel_b = np.einsum("nij,nj->ni", np.swapaxes(root_rotation, 1, 2), root_lin_vel)
        root_ang_vel_b = np.einsum("nij,nj->ni", np.swapaxes(root_rotation, 1, 2), root_ang_vel)
        self.reference_quat = root_quat
        self.features = np.concatenate(
            [self.joint_pos, self.joint_vel, _rot6d(root_quat), root_lin_vel_b, root_ang_vel_b],
            axis=1,
        )
        if self.joint_pos.shape[1] != 29 or self.features.shape[1] != 70:
            raise ValueError(f"Unexpected GRIT motion shape: joints={self.joint_pos.shape}, features={self.features.shape}")

        self.history = [
            np.zeros((HISTORY_LENGTH, 3), dtype=np.float32),
            np.zeros((HISTORY_LENGTH, 3), dtype=np.float32),
            np.zeros((HISTORY_LENGTH, 6), dtype=np.float32),
            np.zeros((HISTORY_LENGTH, 29), dtype=np.float32),
            np.zeros((HISTORY_LENGTH, 29), dtype=np.float32),
            np.zeros((HISTORY_LENGTH, 29), dtype=np.float32),
        ]
        self.last_action = np.zeros(29, dtype=np.float32)
        self.reference_index = 0
        self.transition_start = GRIT_DEFAULT_Q.copy()
        self.transition_steps = 100
        self.transition_index = 0
        self.reset(GRIT_DEFAULT_Q, np.zeros(35, dtype=np.float32), np.array([1, 0, 0, 0], dtype=np.float32))
        print(f"[GRIT] Loaded inference policy: {policy_path.name}; reference={motion_path.name} ({len(self.joint_pos)} frames @ {self.fps:g} Hz)")

    @property
    def default_qpos(self) -> np.ndarray:
        return GRIT_DEFAULT_Q.copy()

    def reset(self, qpos: np.ndarray, qvel: np.ndarray, root_quat: np.ndarray) -> None:
        self.reference_index = 0
        self.last_action[:] = 0.0
        self.transition_start = np.asarray(qpos, dtype=np.float32).copy()
        self.transition_index = 0
        for history in self.history:
            history[:] = 0.0
        for _ in range(HISTORY_LENGTH):
            self._push_history(qpos, qvel, root_quat)

    def _push_history(self, qpos: np.ndarray, qvel: np.ndarray, root_quat: np.ndarray) -> None:
        idx = min(self.reference_index, len(self.features) - 1)
        reference_quat = self.reference_quat[idx]
        robot_rotation = _rotation_matrix_wxyz(np.asarray(root_quat, dtype=np.float32).reshape(1, 4))[0]
        reference_rotation = _rotation_matrix_wxyz(reference_quat.reshape(1, 4))[0]
        relative_rot6d = (robot_rotation.T @ reference_rotation)[:, :2].reshape(-1).astype(np.float32)
        frames = (
            np.asarray(qvel[3:6], dtype=np.float32),
            _projected_gravity(np.asarray(root_quat, dtype=np.float32).reshape(1, 4))[0],
            relative_rot6d,
            np.asarray(qpos, dtype=np.float32) - GRIT_DEFAULT_Q,
            np.asarray(qvel[6:35], dtype=np.float32),
            self.last_action,
        )
        for history, frame in zip(self.history, frames):
            history[:-1] = history[1:]
            history[-1] = frame

    def infer(self, qpos: np.ndarray, qvel: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
        self._push_history(qpos, qvel, root_quat)
        indices = np.clip(self.reference_index + CONTEXT_OFFSETS, 0, len(self.features) - 1)
        reference_context = self.features[indices][None, :, :].astype(np.float32)
        proprio_history = np.concatenate([history.reshape(-1) for history in self.history])[None, :].astype(np.float32)
        result = self.session.run(
            ["actions"],
            {
                "reference_context": reference_context,
                "proprio_history": proprio_history,
            },
        )[0]
        raw_action = np.asarray(result[0], dtype=np.float32)
        if raw_action.shape != (29,) or not np.all(np.isfinite(raw_action)):
            raise ValueError(f"Invalid GRIT action shape/value: {raw_action.shape}")
        self.last_action[:] = raw_action
        target = raw_action * GRIT_ACTION_SCALE + GRIT_DEFAULT_Q
        if self.transition_index < self.transition_steps:
            alpha = (self.transition_index + 1) / self.transition_steps
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)
            target = self.transition_start * (1.0 - alpha) + target * alpha
            self.transition_index += 1
        self.reference_index = (self.reference_index + 1) % len(self.features)
        return target.astype(np.float32)
