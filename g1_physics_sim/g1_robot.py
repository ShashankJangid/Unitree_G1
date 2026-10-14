from __future__ import annotations
import math
import pathlib
from typing import Dict, List, Optional, Tuple
import numpy as np
import mujoco
try:
    from .g1_constants import (    OFFICIAL_KD, OFFICIAL_KP, JOINT_NAMES_29DOF,
    NOMINAL_STAND_Q, ZERO_POSE_Q, NUM_BODY_MOTORS, TORQUE_LIMITS,
    L_WRIST_YAW, R_WRIST_YAW, ARM_SDK_ENABLE, LEG_JOINTS,
    LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, WAIST_JOINTS,
    L_ANKLE_PITCH, R_ANKLE_PITCH, L_ANKLE_ROLL, R_ANKLE_ROLL,
    L_HIP_PITCH, R_HIP_PITCH, L_HIP_ROLL, R_HIP_ROLL,
    JOINT_LIMITS_MIN, JOINT_LIMITS_MAX,
    ELASTIC_BAND_HEIGHT, MODE_PR, MODE_AB,
    SELF_COLLISION_DIST, PHYSICS_DT)
except ImportError:
    from g1_constants import (OFFICIAL_KD, OFFICIAL_KP, JOINT_NAMES_29DOF,
NOMINAL_STAND_Q, ZERO_POSE_Q, NUM_BODY_MOTORS, TORQUE_LIMITS,
L_WRIST_YAW, R_WRIST_YAW, ARM_SDK_ENABLE, LEG_JOINTS,
LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, WAIST_JOINTS,
L_ANKLE_PITCH, R_ANKLE_PITCH, L_ANKLE_ROLL, R_ANKLE_ROLL,
L_HIP_PITCH, R_HIP_PITCH, L_HIP_ROLL, R_HIP_ROLL,
JOINT_LIMITS_MIN, JOINT_LIMITS_MAX,
ELASTIC_BAND_HEIGHT, MODE_PR, MODE_AB,
SELF_COLLISION_DIST, PHYSICS_DT)

DEFAULT_SCENE = 'models/g1/g1_physics_scene.xml'
DEFAULT_SCENE_WITH_HAND = 'models/g1/scene_29dof_with_hand.xml'


class G1Robot:
    """
    MuJoCo model of the G1 29-DOF body.

    The body is always exactly 29 actuators. Dexterous hands are modelled by a
    separate XML if present, but they are NOT appended to the 0..28 body index
    space - that index space is defined by the real robot's LowCmd_ layout and
    must never be renumbered.
    """

    def __init__(
        self,
        xml_path: Optional[str] = None,
        fix_base: bool = False,
        num_motors: int = NUM_BODY_MOTORS,
        enable_elastic_band: bool = False,
        elastic_band_height: float = ELASTIC_BAND_HEIGHT,
        with_hands: bool = False,
    ):
        if num_motors != NUM_BODY_MOTORS:
            raise ValueError(
                f'G1 body is exactly {NUM_BODY_MOTORS} motors, got {num_motors}. '
                'Hands are controlled separately, not appended to the body array.'
            )
        self.num_motors = NUM_BODY_MOTORS
        self.with_hands = with_hands
        self.fix_base = fix_base
        self.enable_elastic_band = enable_elastic_band
        self.elastic_band_height = elastic_band_height
        self.elastic_band_active = enable_elastic_band
        self.mode_pr = MODE_PR
        self.is_walking = False
        self.walk_vx = 0.0
        self.walk_vy = 0.0
        self.walk_wz = 0.0
        self.target_yaw = 0.0

        base = pathlib.Path(__file__).resolve().parent
        if xml_path is None:
            xml_path = str(base / (DEFAULT_SCENE_WITH_HAND if with_hands else DEFAULT_SCENE))

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = PHYSICS_DT

        self._qpos_adr = np.zeros(self.num_motors, dtype=np.int32)
        self._dof_adr = np.zeros(self.num_motors, dtype=np.int32)
        self._actuator_id = np.zeros(self.num_motors, dtype=np.int32)

        for i, name in enumerate(JOINT_NAMES_29DOF):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise RuntimeError(f"Joint '{name}' not found in MuJoCo model at {xml_path}")
            self._qpos_adr[i] = int(self.model.jnt_qposadr[jid])
            self._dof_adr[i] = int(self.model.jnt_dofadr[jid])
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace('_joint', ''))
            self._actuator_id[i] = aid

        self._q_min = np.zeros(self.num_motors, dtype=np.float32)
        self._q_max = np.zeros(self.num_motors, dtype=np.float32)
        for i, name in enumerate(JOINT_NAMES_29DOF):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if self.model.jnt_limited[jid]:
                self._q_min[i] = float(self.model.jnt_range[jid][0])
                self._q_max[i] = float(self.model.jnt_range[jid][1])
            else:
                self._q_min[i] = -np.pi
                self._q_max[i] = np.pi

        self.left_hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_wrist_yaw_link')
        self.right_hand_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_wrist_yaw_link')
        self.pelvis_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        self.target_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_marker')
        self.target_mid = -1
        if self.target_bid != -1:
            self.target_mid = int(self.model.body_mocapid[self.target_bid])

        self.left_shoulder_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_shoulder_pitch_link')
        self.right_shoulder_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_shoulder_pitch_link')

        self.kp = OFFICIAL_KP[:self.num_motors].copy()
        self.kd = OFFICIAL_KD[:self.num_motors].copy()
        for idx in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS:
            self.kp[idx] = 120.0
            self.kd[idx] = 3.0
        for idx in LEG_JOINTS:
            self.kp[idx] = 220.0
            self.kd[idx] = 10.0

        self.foot_gids = set()
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'ground_floor')
        if self.floor_gid == -1:
            self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self.floor_original_matid = int(self.model.geom_matid[self.floor_gid]) if self.floor_gid != -1 else -1
        self.is_partition_plane = True
        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname and ('ankle' in gname or 'foot' in gname):
                self.foot_gids.add(gid)

        self._elastic_band_eqid = -1
        if self.enable_elastic_band:
            self._setup_elastic_band()

        self.reset()

    def _setup_elastic_band(self) -> None:
        """Locate a pre-built elastic-band equality constraint in the MJCF."""
        self._elastic_band_eqid = -1
        for eq_id in range(self.model.neq):
            eq_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id)
            if eq_name and 'elastic' in eq_name.lower():
                self._elastic_band_eqid = eq_id
                return
        if self.enable_elastic_band:
            print('[G1Robot] No elastic-band equality constraint in scene; '
                  'free-standing will collapse without a balance controller.')

    def set_mode_pr(self, use_pr: bool) -> None:
        """
        Ankle/waist actuation mode.
        PR (0) = series pitch/roll layout, AB (1) = parallel A/B layout.
        The underlying MuJoCo joints are identical; the flag only documents the
        intended interpretation to match LowCmd_.mode_pr.
        """
        self.mode_pr = MODE_PR if use_pr else MODE_AB

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
        init_z = 0.793 if self.fix_base else 0.77417
        self.data.qpos[0:3] = [0.0, 0.0, init_z]
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
        """Applied actuator force per body motor (always length 29)."""
        out = np.zeros(self.num_motors, dtype=np.float32)
        for i in range(self.num_motors):
            aid = self._actuator_id[i]
            if aid != -1:
                out[i] = self.data.actuator_force[aid]
        return out

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

    def set_plane_type(self, partition: bool):
        if self.floor_gid == -1:
            return
        self.is_partition_plane = partition
        if partition:
            self.model.geom_matid[self.floor_gid] = self.floor_original_matid
            self.model.geom_rgba[self.floor_gid] = [0.5, 0.5, 0.5, 1.0]
        else:
            self.model.geom_matid[self.floor_gid] = -1
            self.model.geom_rgba[self.floor_gid] = [0.28, 0.30, 0.34, 1.0]

    def toggle_plane_type(self) -> bool:
        self.set_plane_type(not self.is_partition_plane)
        return self.is_partition_plane

    def compute_pd_torques(
        self,
        q_des: np.ndarray,
        dq_des: Optional[np.ndarray] = None,
        tau_ff: Optional[np.ndarray] = None
    ) -> np.ndarray:
        q_des = self.clamp_to_limits(np.asarray(q_des, dtype=np.float32)[:self.num_motors])
        q_act = self.get_joint_positions()
        dq_act = self.get_joint_velocities()

        if dq_des is None:
            dq_des = np.zeros(self.num_motors, dtype=np.float32)
        if tau_ff is None:
            tau_ff = np.zeros(self.num_motors, dtype=np.float32)

        tau_ff = np.clip(np.asarray(tau_ff, dtype=np.float32)[:self.num_motors],
                         -TORQUE_LIMITS[:self.num_motors], TORQUE_LIMITS[:self.num_motors])
        torques = self.kp * (q_des - q_act) + self.kd * (dq_des[:self.num_motors] - dq_act) + tau_ff
        return np.clip(torques, -TORQUE_LIMITS[:self.num_motors], TORQUE_LIMITS[:self.num_motors])

    def step_physics(
        self,
        q_des: np.ndarray,
        dq_des: Optional[np.ndarray] = None,
        tau_ff: Optional[np.ndarray] = None,
        write_ctrl: bool = True
    ) -> np.ndarray:
        """
        Advance physics one 500 Hz tick.

        write_ctrl=False leaves data.ctrl alone so an external controller
        (e.g. G1LocomotionEngine, which computes its own high-fidelity actuator
        torques) can own it. Otherwise this PD overwrites the external torque
        every step and the external policy becomes a silent no-op.
        """
        cmd = np.array(q_des, dtype=np.float32)[:self.num_motors].copy()
        if not self.fix_base and self.pelvis_bid != -1:
            target_z = 0.774
            kp_z = 1800.0
            kd_z = 250.0
            kp_att = 450.0
            kd_att = 60.0

            fz = kp_z * (target_z - self.data.qpos[2]) - kd_z * self.data.qvel[2] + 35.0 * 9.81

            w, x, y, z = float(self.data.qpos[3]), float(self.data.qpos[4]), float(self.data.qpos[5]), float(self.data.qpos[6])
            current_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

            if self.is_walking:
                self.target_yaw = (self.target_yaw + self.walk_wz * PHYSICS_DT + math.pi) % (2.0 * math.pi) - math.pi
                cos_h = math.cos(current_yaw)
                sin_h = math.sin(current_yaw)
                world_vx = cos_h * self.walk_vx - sin_h * self.walk_vy
                world_vy = sin_h * self.walk_vx + cos_h * self.walk_vy
                fx = 70.0 * (world_vx - self.data.qvel[0])
                fy = 70.0 * (world_vy - self.data.qvel[1])
            else:
                self.target_yaw = current_yaw
                fx = 600.0 * (0.0 - self.data.qpos[0]) - 120.0 * self.data.qvel[0]
                fy = 600.0 * (0.0 - self.data.qpos[1]) - 120.0 * self.data.qvel[1]

            R = self.data.xmat[self.pelvis_bid].reshape(3, 3)
            cy = math.cos(self.target_yaw)
            sy = math.sin(self.target_yaw)
            R_des = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            M = R_des @ R.T
            ex = 0.5 * (M[2, 1] - M[1, 2])
            ey = 0.5 * (M[0, 2] - M[2, 0])
            ez = 0.5 * (M[1, 0] - M[0, 1])

            ww = R @ self.data.qvel[3:6]
            des_wz = self.walk_wz if self.is_walking else 0.0

            tx = kp_att * ex - kd_att * ww[0]
            ty = kp_att * ey - kd_att * ww[1]
            tz = 160.0 * ez + 35.0 * (des_wz - ww[2]) if self.is_walking else 200.0 * ez - 35.0 * ww[2]

            self.data.xfrc_applied[self.pelvis_bid, 0] = np.clip(fx, -120.0, 120.0)
            self.data.xfrc_applied[self.pelvis_bid, 1] = np.clip(fy, -120.0, 120.0)
            self.data.xfrc_applied[self.pelvis_bid, 2] = np.clip(fz, 0.0, 600.0)
            self.data.xfrc_applied[self.pelvis_bid, 3] = np.clip(tx, -120.0, 120.0)
            self.data.xfrc_applied[self.pelvis_bid, 4] = np.clip(ty, -120.0, 120.0)
            self.data.xfrc_applied[self.pelvis_bid, 5] = np.clip(tz, -60.0, 60.0)

        torques = self.compute_pd_torques(cmd, dq_des, tau_ff)

        if write_ctrl:
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

    def check_self_collision(self, threshold: float = SELF_COLLISION_DIST) -> Tuple[bool, float]:
        max_penetration = 0.0
        has_collision = False
        for i in range(self.data.ncon):
            if self.data.contact[i].dist < threshold:
                has_collision = True
                max_penetration = max(max_penetration, -float(self.data.contact[i].dist))
        return has_collision, max_penetration

    def get_joint_limits(self) -> Tuple[np.ndarray, np.ndarray]:
        """Limits read from the MJCF at construction time."""
        return self._q_min.copy(), self._q_max.copy()

    def clamp_to_limits(self, q: np.ndarray, margin: float = 0.03) -> np.ndarray:
        q = np.asarray(q, dtype=np.float32)
        lo = self._q_min + margin
        hi = self._q_max - margin
        hi = np.maximum(hi, lo)
        return np.clip(q, lo, hi)

    def get_leg_state(self) -> Tuple[np.ndarray, np.ndarray]:
        q = self.get_joint_positions()[LEG_JOINTS]
        dq = self.get_joint_velocities()[LEG_JOINTS]
        return q, dq

    def get_arm_state(self, right_arm: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        indices = RIGHT_ARM_JOINTS if right_arm else LEFT_ARM_JOINTS
        q = self.get_joint_positions()[indices]
        dq = self.get_joint_velocities()[indices]
        return q, dq

    def get_waist_state(self) -> Tuple[np.ndarray, np.ndarray]:
        q = self.get_joint_positions()[WAIST_JOINTS]
        dq = self.get_joint_velocities()[WAIST_JOINTS]
        return q, dq

    def set_motor_modes(self, modes: np.ndarray):
        """Set motor modes (for hardware bridge compatibility)."""
        pass
