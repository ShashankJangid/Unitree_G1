from __future__ import annotations
import pathlib
from typing import Dict, List, Optional, Tuple
import numpy as np
import mujoco
from g1_constants import (
    DEFAULT_KD, DEFAULT_KP, JOINT_NAMES_29DOF, JOINT_NAMES_43DOF,
    NOMINAL_STAND_Q, ZERO_POSE_Q, NUM_MOTORS, NUM_BODY_MOTORS, TORQUE_LIMITS,
    L_WRIST_YAW, R_WRIST_YAW, ARM_SDK_ENABLE, LEG_JOINT_INDICES,
    LEFT_ARM_INDICES, RIGHT_ARM_INDICES, WAIST_INDICES,
    JOINT_LIMITS_MIN, JOINT_LIMITS_MAX, ELASTIC_BAND_STIFFNESS, ELASTIC_BAND_DAMPING
)

class G1Robot:

    def __init__(
        self,
        xml_path: Optional[str] = None,
        fix_base: bool = False,
        num_motors: int = NUM_BODY_MOTORS,
        enable_elastic_band: bool = True,
        elastic_band_height: float = 0.793
    ):
        self.num_motors = num_motors
        self.fix_base = fix_base
        self.enable_elastic_band = enable_elastic_band
        self.elastic_band_height = elastic_band_height
        self.elastic_band_active = enable_elastic_band

        if xml_path is None:
            if num_motors == NUM_MOTORS:
                xml_path = str(
                    pathlib.Path(__file__).resolve().parent / 'models' / 'g1' / 'scene_29dof_with_hand.xml'
                )
            else:
                xml_path = str(
                    pathlib.Path(__file__).resolve().parent / 'models' / 'g1' / 'g1_physics_scene.xml'
                )

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = 0.002

        self._qpos_adr = np.zeros(self.num_motors, dtype=np.int32)
        self._dof_adr = np.zeros(self.num_motors, dtype=np.int32)
        self._actuator_id = np.zeros(self.num_motors, dtype=np.int32)

        joint_names = JOINT_NAMES_43DOF if num_motors == NUM_MOTORS else JOINT_NAMES_29DOF

        for i, name in enumerate(joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise RuntimeError(f"Joint '{name}' not found in MuJoCo model!")
            self._qpos_adr[i] = self.model.jnt_qposadr[jid]
            self._dof_adr[i] = self.model.jnt_dofadr[jid]
            act_name = name.replace('_joint', '')
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            self._actuator_id[i] = aid

        self.left_hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_wrist_yaw_link')
        self.right_hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_yaw_link')
        self.pelvis_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.target_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_marker')
        self.target_mid = -1
        if self.target_bid != -1:
            self.target_mid = int(self.model.body_mocapid[self.target_bid])

        self.left_shoulder_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_shoulder_pitch_link')
        self.right_shoulder_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_shoulder_pitch_link')

        self.kp = DEFAULT_KP[:self.num_motors].copy()
        self.kd = DEFAULT_KD[:self.num_motors].copy()

        self.foot_gids = set()
        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname and ('ankle' in gname or 'foot' in gname):
                self.foot_gids.add(gid)

        self._elastic_band_gid = -1
        if self.enable_elastic_band:
            self._setup_elastic_band()

        self.reset()

    def _setup_elastic_band(self):
        """Create a virtual elastic band constraint to suspend the robot."""
        try:
            for eq_id in range(self.model.neq):
                eq_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id)
                if eq_name and 'elastic' in eq_name.lower():
                    self._elastic_band_eqid = eq_id
                    return

            self._elastic_band_eqid = -1
        except Exception:
            self._elastic_band_eqid = -1

    def toggle_elastic_band(self, active: Optional[bool] = None) -> bool:
        """Toggle or set the elastic band state."""
        if active is None:
            self.elastic_band_active = not self.elastic_band_active
        else:
            self.elastic_band_active = active

        if self._elastic_band_eqid != -1:
            self.model.eq_active[self._elastic_band_eqid] = int(self.elastic_band_active)

        return self.elastic_band_active

    def set_elastic_band_height(self, height: float):
        """Adjust the elastic band target height."""
        self.elastic_band_height = height
        if self._elastic_band_eqid != -1:
            self.model.eq_data[self._elastic_band_eqid, 0] = height

    def reset(self, initial_q: Optional[np.ndarray] = None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, self.elastic_band_height if self.elastic_band_active else 0.793]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        q_des = initial_q if initial_q is not None else NOMINAL_STAND_Q[:self.num_motors]
        for i in range(min(self.num_motors, len(q_des))):
            self.data.qpos[self._qpos_adr[i]] = float(q_des[i])

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_joint_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self._qpos_adr], dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        return np.array([self.data.qvel[adr] for adr in self._dof_adr], dtype=np.float32)

    def get_joint_torques(self) -> np.ndarray:
        return np.array([self.data.actuator_force[aid] for aid in self._actuator_id if aid != -1], dtype=np.float32)

    def get_hand_pos(self, right_arm: bool = False) -> np.ndarray:
        bid = self.right_hand_bid if right_arm else self.left_hand_bid
        return np.array(self.data.xpos[bid], dtype=np.float32)

    def get_hand_quat(self, right_arm: bool = False) -> np.ndarray:
        bid = self.right_hand_bid if right_arm else self.left_hand_bid
        return np.array(self.data.xquat[bid], dtype=np.float32)

    def get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array(self.data.qpos[0:3], dtype=np.float32)
        quat = np.array(self.data.qpos[3:7], dtype=np.float32)
        return (pos, quat)

    def get_base_vel(self) -> Tuple[np.ndarray, np.ndarray]:
        linvel = np.array(self.data.qvel[0:3], dtype=np.float32)
        angvel = np.array(self.data.qvel[3:6], dtype=np.float32)
        return (linvel, angvel)

    def get_imu_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get IMU data: quaternion, angular velocity, linear acceleration."""
        imu_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'imu_link')
        if imu_bid == -1:
            imu_bid = self.pelvis_bid
        quat = np.array(self.data.xquat[imu_bid], dtype=np.float32)
        angvel = np.array(self.data.cvel[imu_bid][3:6], dtype=np.float32)
        linacc = np.array(self.data.cacc[imu_bid][0:3], dtype=np.float32)
        return quat, angvel, linacc

    def get_foot_contacts(self) -> Tuple[bool, bool]:
        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            if geom1 in self.foot_gids or geom2 in self.foot_gids:
                if geom1 in self.foot_gids:
                    gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
                else:
                    gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
                if gname and 'left' in gname:
                    left_contact = True
                if gname and 'right' in gname:
                    right_contact = True
        return left_contact, right_contact

    def set_target_marker(self, pos: np.ndarray):
        if self.target_mid != -1:
            self.data.mocap_pos[self.target_mid] = pos

    def compute_pd_torques(
        self,
        q_des: np.ndarray,
        dq_des: Optional[np.ndarray] = None,
        tau_ff: Optional[np.ndarray] = None
    ) -> np.ndarray:
        q_act = self.get_joint_positions()
        dq_act = self.get_joint_velocities()

        if dq_des is None:
            dq_des = np.zeros(self.num_motors, dtype=np.float32)
        if tau_ff is None:
            tau_ff = np.zeros(self.num_motors, dtype=np.float32)

        torques = self.kp * (q_des - q_act) + self.kd * (dq_des - dq_act) + tau_ff
        torques = np.clip(torques, -TORQUE_LIMITS[:self.num_motors], TORQUE_LIMITS[:self.num_motors])
        return torques

    def step_physics(
        self,
        q_des: np.ndarray,
        dq_des: Optional[np.ndarray] = None,
        tau_ff: Optional[np.ndarray] = None
    ) -> np.ndarray:
        torques = self.compute_pd_torques(q_des, dq_des, tau_ff)

        for i in range(self.num_motors):
            aid = self._actuator_id[i]
            if aid != -1:
                self.data.ctrl[aid] = torques[i]

        if self.fix_base:
            self.data.qpos[0:2] = 0.0
            self.data.qpos[2] = 0.793
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            self.data.qvel[0:6] = 0.0

        mujoco.mj_step(self.model, self.data)
        return torques

    def apply_action(self, q_des: np.ndarray, kp: Optional[np.ndarray] = None, kd: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply joint position targets with optional per-joint KP/KD."""
        if kp is not None:
            self.kp[:len(kp)] = kp
        if kd is not None:
            self.kd[:len(kd)] = kd
        return self.step_physics(q_des)

    def check_self_collision(self, threshold: float = -0.002) -> Tuple[bool, float]:
        max_penetration = 0.0
        has_collision = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.dist < threshold:
                has_collision = True
                max_penetration = max(max_penetration, -contact.dist)
        return has_collision, max_penetration

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        return JOINT_LIMITS_MIN[:self.num_motors].copy(), JOINT_LIMITS_MAX[:self.num_motors].copy()

    def clamp_to_limits(self, q: np.ndarray, margin: float = 0.03) -> np.ndarray:
        q_min, q_max = self.get_joint_limits()
        return np.clip(q, q_min + margin, q_max - margin)

    def get_leg_state(self) -> Tuple[np.ndarray, np.ndarray]:
        q = self.get_joint_positions()[LEG_JOINT_INDICES]
        dq = self.get_joint_velocities()[LEG_JOINT_INDICES]
        return q, dq

    def get_arm_state(self, right_arm: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        indices = RIGHT_ARM_INDICES if right_arm else LEFT_ARM_INDICES
        q = self.get_joint_positions()[indices]
        dq = self.get_joint_velocities()[indices]
        return q, dq

    def get_waist_state(self) -> Tuple[np.ndarray, np.ndarray]:
        q = self.get_joint_positions()[WAIST_INDICES]
        dq = self.get_joint_velocities()[WAIST_INDICES]
        return q, dq

    def set_motor_modes(self, modes: np.ndarray):
        """Set motor modes (for hardware bridge compatibility)."""
        pass
