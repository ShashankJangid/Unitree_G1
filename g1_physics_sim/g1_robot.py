from __future__ import annotations
import pathlib
from typing import Dict, List, Optional, Tuple
import numpy as np
import mujoco
from g1_constants import DEFAULT_KD, DEFAULT_KP, JOINT_NAMES_29DOF, NOMINAL_STAND_Q, NUM_MOTORS, TORQUE_LIMITS, L_WRIST_YAW, R_WRIST_YAW

class G1Robot:

    def __init__(self, xml_path: Optional[str]=None, fix_base: bool=False):
        if xml_path is None:
            xml_path = str(pathlib.Path(__file__).resolve().parent / 'models' / 'g1' / 'g1_physics_scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.fix_base = fix_base
        self.model.opt.timestep = 0.002
        self._qpos_adr = np.zeros(NUM_MOTORS, dtype=np.int32)
        self._dof_adr = np.zeros(NUM_MOTORS, dtype=np.int32)
        self._actuator_id = np.zeros(NUM_MOTORS, dtype=np.int32)
        for i, name in enumerate(JOINT_NAMES_29DOF):
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
        self.kp = DEFAULT_KP.copy()
        self.kd = DEFAULT_KD.copy()
        self.foot_gids = set()
        for gid in range(self.model.ngeom):
            gname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if gname and ('ankle' in gname or 'foot' in gname):
                self.foot_gids.add(gid)
        self.reset()

    def reset(self, initial_q: Optional[np.ndarray]=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = [0.0, 0.0, 0.793]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        q_des = initial_q if initial_q is not None else NOMINAL_STAND_Q
        for i in range(NUM_MOTORS):
            self.data.qpos[self._qpos_adr[i]] = float(q_des[i])
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_joint_positions(self) -> np.ndarray:
        return np.array([self.data.qpos[adr] for adr in self._qpos_adr], dtype=np.float32)

    def get_joint_velocities(self) -> np.ndarray:
        return np.array([self.data.qvel[adr] for adr in self._dof_adr], dtype=np.float32)

    def get_hand_pos(self, right_arm: bool=False) -> np.ndarray:
        bid = self.right_hand_bid if right_arm else self.left_hand_bid
        return np.array(self.data.xpos[bid], dtype=np.float32)

    def get_base_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.array(self.data.qpos[0:3], dtype=np.float32)
        quat = np.array(self.data.qpos[3:7], dtype=np.float32)
        return (pos, quat)

    def set_target_marker(self, pos: np.ndarray):
        if self.target_mid != -1:
            self.data.mocap_pos[self.target_mid] = pos

    def compute_pd_torques(self, q_des: np.ndarray, dq_des: Optional[np.ndarray]=None, tau_ff: Optional[np.ndarray]=None) -> np.ndarray:
        q_act = self.get_joint_positions()
        dq_act = self.get_joint_velocities()
        if dq_des is None:
            dq_des = np.zeros(NUM_MOTORS, dtype=np.float32)
        if tau_ff is None:
            tau_ff = np.zeros(NUM_MOTORS, dtype=np.float32)
        torques = self.kp * (q_des - q_act) + self.kd * (dq_des - dq_act) + tau_ff
        torques = np.clip(torques, -TORQUE_LIMITS, TORQUE_LIMITS)
        return torques

    def step_physics(self, q_des: np.ndarray, dq_des: Optional[np.ndarray]=None, tau_ff: Optional[np.ndarray]=None) -> np.ndarray:
        torques = self.compute_pd_torques(q_des, dq_des, tau_ff)
        for i in range(NUM_MOTORS):
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
