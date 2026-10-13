from __future__ import annotations
import json
import pathlib
from typing import Optional, Tuple
import numpy as np
import mujoco

try:
    import onnxruntime as ort
except ImportError:
    ort = None

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
        self.decimation = 4
        self.step_count = 0

        self.base_dir = pathlib.Path(__file__).resolve().parent
        self.policies_dir = self.base_dir / "policies"

        self.cfg_file = self.policies_dir / "model_config.json"
        with open(self.cfg_file, "r") as f:
            self.config = json.load(f)

        self.joint_names = self.config["joint_names"]
        self.num_joints = len(self.joint_names)
        self.default_joint_pos = np.array([self.config["default_joint_pos"][n] for n in self.joint_names], dtype=np.float32)
        self.action_scales = np.array([self.config["action_scales"][n] for n in self.joint_names], dtype=np.float32)

        arm_patterns = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
        self.arm_indices = [i for i, n in enumerate(self.joint_names) if any(p in n for p in arm_patterns)]

        self._build_pd_gains()
        self._apply_model_armature()

        self.act_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n.replace("_joint", "")) for n in self.joint_names]
        self.qpos_indices = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.joint_names]
        self.qvel_indices = [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in self.joint_names]

        self.walker_sess = None
        self.rotator_sess = None
        if ort is not None:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            w_path = self.policies_dir / "walker.onnx"
            r_path = self.policies_dir / "rotator.onnx"
            if w_path.exists():
                self.walker_sess = ort.InferenceSession(str(w_path), opts, providers=["CPUExecutionProvider"])
            if r_path.exists():
                self.rotator_sess = ort.InferenceSession(str(r_path), opts, providers=["CPUExecutionProvider"])

        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self.target_joint_pos = self.default_joint_pos.copy()

    def _build_pd_gains(self):
        S5020, D5020, E5020 = 14.2506, 0.9072, 25.0
        S7520_14, D7520_14, E7520_14 = 40.1792, 2.5579, 88.0
        S7520_22, D7520_22, E7520_22 = 99.0984, 6.3088, 139.0
        S4010, D4010, E4010 = 16.7783, 1.0681, 5.0

        self.kp = np.zeros(self.num_joints, dtype=np.float32)
        self.kd = np.zeros(self.num_joints, dtype=np.float32)
        self.effort_limit = np.zeros(self.num_joints, dtype=np.float32)

        for i, name in enumerate(self.joint_names):
            if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
                self.kp[i], self.kd[i], self.effort_limit[i] = S5020, D5020, E5020
            elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
                self.kp[i], self.kd[i], self.effort_limit[i] = S7520_14, D7520_14, E7520_14
            elif "hip_roll" in name or "knee" in name:
                self.kp[i], self.kd[i], self.effort_limit[i] = S7520_22, D7520_22, E7520_22
            elif "wrist_pitch" in name or "wrist_yaw" in name:
                self.kp[i], self.kd[i], self.effort_limit[i] = S4010, D4010, E4010
            elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
                self.kp[i], self.kd[i], self.effort_limit[i] = S5020 * 2, D5020 * 2, E5020 * 2
            else:
                self.kp[i], self.kd[i], self.effort_limit[i] = S5020, D5020, E5020

    def _apply_model_armature(self):
        ARM_5020 = 0.00360972
        ARM_7520_14 = 0.01017752
        ARM_7520_22 = 0.02510192
        ARM_4010 = 0.00425000
        ARM_2x5020 = 0.00721945

        for i, name in enumerate(self.joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                dof = self.model.jnt_dofadr[jid]
                if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
                    self.model.dof_armature[dof] = ARM_5020
                elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
                    self.model.dof_armature[dof] = ARM_7520_14
                elif "hip_roll" in name or "knee" in name:
                    self.model.dof_armature[dof] = ARM_7520_22
                elif "wrist_pitch" in name or "wrist_yaw" in name:
                    self.model.dof_armature[dof] = ARM_4010
                elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
                    self.model.dof_armature[dof] = ARM_2x5020
                else:
                    self.model.dof_armature[dof] = ARM_5020

    @staticmethod
    def _quat_inv(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        w, xyz = quat[0], quat[1:4]
        t = np.cross(xyz, vec) * 2.0
        return vec - w * t + np.cross(xyz, t)

    def reset(self):
        self.step_count = 0
        self.last_action[:] = 0.0
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_z = 0.0
        self.target_joint_pos = self.default_joint_pos.copy()

    def set_velocity_command(self, vx: float, vy: float, vyaw: float):
        self.lin_vel_x = float(np.clip(vx, -1.0, 1.5))
        self.lin_vel_y = float(np.clip(vy, -0.6, 0.6))
        self.ang_vel_z = float(np.clip(vyaw, -1.0, 1.0))

    def step(self):
        if not self.is_active or self.walker_sess is None:
            return

        if self.step_count % self.decimation == 0:
            quat = self.data.qpos[3:7]
            lin_vel = self._quat_inv(quat, self.data.qvel[:3])
            ang_vel = self.data.qvel[3:6]
            proj_g = self._quat_inv(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))

            j_pos = np.zeros(self.num_joints, dtype=np.float32)
            j_vel = np.zeros(self.num_joints, dtype=np.float32)
            for i in range(self.num_joints):
                j_pos[i] = self.data.qpos[self.qpos_indices[i]] - self.default_joint_pos[i]
                j_vel[i] = self.data.qvel[self.qvel_indices[i]]

            cmd = np.array([self.lin_vel_x, self.lin_vel_y, self.ang_vel_z], dtype=np.float32)
            obs = np.concatenate([lin_vel, ang_vel, proj_g, j_pos, j_vel, self.last_action, cmd]).reshape(1, -1).astype(np.float32)

            use_rotator = abs(self.ang_vel_z) > 0.1 and abs(self.lin_vel_x) < 0.1 and self.rotator_sess is not None
            sess = self.rotator_sess if use_rotator else self.walker_sess

            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            action = sess.run([out_name], {in_name: obs})[0][0]

            self.target_joint_pos = self.default_joint_pos + action * self.action_scales
            for idx in self.arm_indices:
                self.target_joint_pos[idx] = self.default_joint_pos[idx]
            self.last_action = action.copy()

        for i, aid in enumerate(self.act_ids):
            if aid >= 0:
                q_cur = self.data.qpos[self.qpos_indices[i]]
                dq_cur = self.data.qvel[self.qvel_indices[i]]
                tau = self.kp[i] * (self.target_joint_pos[i] - q_cur) - self.kd[i] * dq_cur
                tau = np.clip(tau, -self.effort_limit[i], self.effort_limit[i])
                self.data.ctrl[aid] = tau

        self.step_count += 1
