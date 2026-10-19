"""Inference-only adapter for Unitree RL Lab's Gangnam Style G1 policy.

Deployment contract: upstream State_Mimic.cpp and the unmodified deploy.yaml
in models/gangnam. Joint defaults/actions use breadth-first policy order;
gains and CSV joint samples use depth-first motor/XML order.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ASSET_HASHES = {
    "policy.onnx": "c09a6088fbf85aecd96b1fa89528d7462d4a571a981408d5ff2afb69f0375b7d",
    "deploy.yaml": "164137b43e8834d7c53828d5697aa993911fb2fca5dcb749e737912daf3ca7bb",
    "motion.csv": "d74f54662546969826a25386a21f429b6b7f8fd4793578c9923ca5ce4e9ee0aa",
}
JOINT_MAP = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
KP = np.array([40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 99.1, 40.2, 99.1, 28.5, 28.5, 40.2, 28.5, 28.5, 14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8, 14.3, 14.3, 14.3, 14.3, 14.3, 16.8, 16.8])
KD = np.array([2.56, 6.31, 2.56, 6.31, 1.81, 1.81, 2.56, 6.31, 2.56, 6.31, 1.81, 1.81, 2.56, 1.81, 1.81, .907, .907, .907, .907, .907, 1.07, 1.07, .907, .907, .907, .907, .907, 1.07, 1.07])
# These two arrays are ALREADY in policy order: do not gather JOINT_MAP again.
DEFAULT = np.array([-.302, -.319, .00124, .000442, .00489, .00191, .00929, .00796, .00546, .672, .67, .2, .202, -.368, -.355, .194, -.196, -.00644, .00976, .00258, -.00029, .605, .596, .00818, .00322, .00293, -.00339, -.00955, -.00715])
SCALE = np.array([.548, .548, .548, .351, .351, .439, .548, .548, .439, .351, .351, .439, .439, .439, .439, .439, .439, .439, .439, .439, .439, .439, .439, .439, .439, .0745, .0745, .0745, .0745])


def _vector(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def _rotation(quat):
    quat = _vector(quat, (4,), "quaternion")
    norm = np.linalg.norm(quat)
    if norm < 1e-9:
        raise ValueError("quaternion must have nonzero norm")
    return Rotation.from_quat(np.roll(quat / norm, -1))


def _yaw(rotation):
    forward = rotation.as_matrix()[:, 0]
    return Rotation.from_euler("z", np.arctan2(forward[1], forward[0]))


class GangnamPolicy:
    dt = .02
    start_time = 5.2
    end_time = 25.5
    duration = 20.3

    def __init__(self, asset_dir: Path = Path(__file__).resolve().parent / "models/gangnam"):
        asset_dir = Path(asset_dir)
        for filename, expected in ASSET_HASHES.items():
            path = asset_dir / filename
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f"Gangnam asset hash mismatch: {filename}")
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(asset_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 154] or inputs[0].type != "tensor(float)":
            raise ValueError("Gangnam ONNX requires float32 observation [1,154]")
        if len(outputs) != 1 or outputs[0].shape != [1, 29] or outputs[0].type != "tensor(float)":
            raise ValueError("Gangnam ONNX requires float32 actions [1,29]")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        motion = np.loadtxt(asset_dir / "motion.csv", delimiter=",", dtype=np.float64)
        if motion.shape != (1942, 36) or not np.all(np.isfinite(motion)):
            raise ValueError("Gangnam motion must be finite [1942,36]")
        self.root_pos = motion[:, :3]
        self.root_quat = motion[:, 3:7].copy()  # xyzw in CSV
        norms = np.linalg.norm(self.root_quat, axis=1)
        if np.any(norms < 1e-9):
            raise ValueError("Gangnam motion contains zero quaternion")
        self.root_quat /= norms[:, None]
        self.joint_pos = motion[:, 7:]
        self.joint_vel = np.empty_like(self.joint_pos)
        self.joint_vel[:-1] = np.diff(self.joint_pos, axis=0) * 60.
        self.joint_vel[-1] = self.joint_vel[-2]
        self.kp, self.kd = KP.copy(), KD.copy()
        self.previous_action = np.zeros(29, dtype=np.float32)
        self.alignment = Rotation.identity()

    def sample(self, time_absolute):
        time_absolute = _vector(time_absolute, (), "time_absolute").item()
        frame = np.clip(time_absolute * 60., 0., len(self.joint_pos) - 1.)
        lo = int(frame)
        hi = min(lo + 1, len(self.joint_pos) - 1)
        alpha = frame - lo
        quat = Slerp([0, 1], Rotation.from_quat(self.root_quat[[lo, hi]]))([alpha])[0] if hi != lo else Rotation.from_quat(self.root_quat[lo])
        return ((1-alpha)*self.root_pos[lo] + alpha*self.root_pos[hi],
                np.roll(quat.as_quat(), 1),
                (1-alpha)*self.joint_pos[lo] + alpha*self.joint_pos[hi],
                (1-alpha)*self.joint_vel[lo] + alpha*self.joint_vel[hi])

    def reset(self, root_quat_wxyz, joint_pos_xml):
        q = _vector(joint_pos_xml, (29,), "joint_pos_xml")
        robot = _rotation(root_quat_wxyz) * Rotation.from_euler("ZXY", q[12:15])
        reference = _rotation(self.sample(self.start_time)[1])
        self.alignment = _yaw(robot) * _yaw(reference).inv()
        self.previous_action[:] = 0

    def infer(self, elapsed_seconds, joint_pos_xml, joint_vel_xml, angular_velocity_body, torso_rotation_world_3x3):
        elapsed = _vector(elapsed_seconds, (), "elapsed_seconds").item()
        q = _vector(joint_pos_xml, (29,), "joint_pos_xml")
        dq = _vector(joint_vel_xml, (29,), "joint_vel_xml")
        ang = _vector(angular_velocity_body, (3,), "angular_velocity_body")
        torso = _vector(torso_rotation_world_3x3, (3, 3), "torso_rotation_world_3x3")
        if not np.allclose(torso.T @ torso, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(torso), 1, atol=1e-4):
            raise ValueError("torso rotation must be a proper rotation matrix")
        _, root_quat, ref_q, ref_dq = self.sample(self.start_time + np.clip(elapsed, 0, self.duration))
        ref_torso = (_rotation(root_quat) * Rotation.from_euler("ZXY", ref_q[12:15])).as_matrix()
        relative = torso.T @ self.alignment.as_matrix() @ ref_torso
        obs = np.concatenate((ref_q[JOINT_MAP], ref_dq[JOINT_MAP], relative[:, :2].reshape(6), ang,
                              q[JOINT_MAP] - DEFAULT, dq[JOINT_MAP], self.previous_action)).astype(np.float32)[None]
        if obs.shape != (1, 154) or not np.all(np.isfinite(obs)):
            raise ValueError("invalid Gangnam observation")
        result = self.session.run([self.output_name], {self.input_name: obs})[0]
        if not isinstance(result, np.ndarray) or result.shape != (1, 29) or result.dtype != np.float32 or not np.all(np.isfinite(result)):
            raise ValueError("invalid Gangnam action output")
        raw = result[0]
        target = np.empty(29, dtype=np.float32)
        target[JOINT_MAP] = raw * SCALE.astype(np.float32) + DEFAULT.astype(np.float32)
        self.previous_action[:] = raw
        return target
