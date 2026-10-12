from __future__ import annotations
import math
import pathlib
from typing import Dict, Tuple
import gymnasium as gym
import numpy as np

class G1ArmReachEnv(gym.Env):
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 25}
    _DISABLE_COLLISION: bool = True

    def __init__(self, render_mode: str | None=None, right_arm: bool=False):
        import mujoco
        self._mujoco = mujoco
        self._render_mode = render_mode or 'none'
        self._training_right = bool(right_arm)
        root = pathlib.Path(__file__).resolve().parent
        xml = root / 'unitree_mujoco' / 'unitree_robots' / 'g1' / 'g1_29dof_with_hand.xml'
        if not xml.exists():
            raise FileNotFoundError(xml)
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self._floor_gid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, 'floor')
        self._floor_colours = [(97 / 255.0, 161 / 255.0, 84 / 255.0, 1.0)]
        self._left_idx = list(range(15, 22))
        self._right_idx = list(range(22, 29))
        self._arm_j = self._right_idx if right_arm else self._left_idx
        right_rest_deg = [25.8, -11.5, -2.9, 28.6, 34.4, -17.2, 14.3]
        self._right_joint_names = ['right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']
        self._right_rest_q = np.array(right_rest_deg, dtype=np.float32) * (math.pi / 180.0)
        self._right_qadr = []
        self._right_dadr = []
        for nm in self._right_joint_names:
            jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid == -1:
                raise RuntimeError(f'Joint {nm} not found in model')
            self._right_qadr.append(int(self.model.jnt_qposadr[jid]))
            self._right_dadr.append(int(self.model.jnt_dofadr[jid]))
        if right_arm:
            self._park_qadr = self._left_qadr = [int(self.model.jnt_qposadr[self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm.replace('right_', 'left_'))]) for nm in self._right_joint_names]
            mirror = np.array([25.8, 11.5, 2.9, 28.6, -34.4, 17.2, -14.3], dtype=np.float32)
            self._park_rest_q = mirror * (math.pi / 180.0)
            self._park_dadr = [int(self.model.jnt_dofadr[self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm.replace('right_', 'left_'))]) for nm in self._right_joint_names]
        else:
            self._park_qadr = self._right_qadr
            self._park_rest_q = self._right_rest_q
            self._park_dadr = self._right_dadr
        self._qadr = [int(self.model.jnt_qposadr[j]) for j in self._arm_j]
        self.goal_bid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, 'goal_body')
        self._goal_mid = -1
        if self.goal_bid != -1:
            self._goal_mid = int(self.model.body_mocapid[self.goal_bid])
        self._waist_qadr = []
        waist_names = ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
        for nm in waist_names:
            jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid != -1:
                self._waist_qadr.append(int(self.model.jnt_qposadr[jid]))
        protect_kw = ('pelvis', 'waist', 'torso', 'chest', 'spine', 'abdomen', 'body', 'head', 'neck', 'hip_', 'thigh', 'knee', 'calf', 'shin', 'leg', 'upperarm', 'lowerarm', 'shoulder')
        arm_prefix = 'right_' if right_arm else 'left_'
        self._protect_gids: set[int] = set()
        self._arm_gids: set[int] = set()
        for gid in range(self.model.ngeom):
            bid = int(self.model.geom_bodyid[gid])
            body_name = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, bid)
            if not body_name:
                continue
            limb_kw = ('upperarm', 'elbow', 'wrist', 'hand', 'forearm')
            if body_name.startswith(arm_prefix) and any((k in body_name for k in limb_kw)):
                self._arm_gids.add(gid)
            if any((k in body_name for k in protect_kw)):
                self._protect_gids.add(gid)
        for gid in self._protect_gids | self._arm_gids:
            self.model.geom_contype[gid] |= 1
            self.model.geom_conaffinity[gid] |= 1
        target_mesh = 'right_hand_palm_link' if right_arm else 'left_hand_palm_link'
        self._palm_gid: int = -1
        self._use_wrist_offset = False
        meshid_field = 'geom_meshid' if hasattr(self.model, 'geom_meshid') else 'geom_dataid'
        for gid in range(self.model.ngeom):
            mid = int(getattr(self.model, meshid_field)[gid])
            if mid == -1:
                continue
            mname = self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_MESH, mid)
            if mname == target_mesh:
                self._palm_gid = gid
                break
        if self._palm_gid == -1:
            self._use_wrist_offset = True
            fallback_body = 'right_wrist_yaw_link' if right_arm else 'left_wrist_yaw_link'
            self._body_id = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_BODY, fallback_body)
            if self._body_id == -1:
                raise RuntimeError(f"Palm mesh '{target_mesh}' not found and fallback body '{fallback_body}' missing")
            self._palm_offset = np.array([0.0415, -0.003 if right_arm else 0.003, 0.0], dtype=np.float32)
        else:
            self._body_id = -1
        if self._palm_gid == -1 and (not self._use_wrist_offset):
            raise RuntimeError(f"Palm geometry '{target_mesh}' not found in the model and no fallback could be configured.")
        if self._palm_gid != -1:
            self._body_id = -1
        shoulder_body_name = 'right_shoulder_pitch_link' if right_arm else 'left_shoulder_pitch_link'
        self._shoulder_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, shoulder_body_name)
        if self._shoulder_bid == -1:
            raise RuntimeError(f'Body {shoulder_body_name} not found in model')
        self.action_space = gym.spaces.Box(-0.05, 0.05, shape=(7,), dtype=np.float32)
        obs_hi = np.array([*[math.pi] * 7, *[10.0] * 7, *[1.5] * 3, *[1.5] * 3, *[1.5] * 3, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-obs_hi, obs_hi, dtype=np.float32)
        self.viewer = None
        if self._render_mode == 'human':
            try:
                from mujoco import viewer as mj_viewer
                self.viewer = mj_viewer.launch_passive(self.model, self.data)
            except Exception:
                print('[warn] MuJoCo viewer not available – falling back to headless')
                self.viewer = None
        self.max_steps = 200
        self._step_count = 0
        if self._floor_gid != -1:
            rgba = self.np_random.choice(self._floor_colours)
            self.model.geom_rgba[self._floor_gid] = rgba

    def _fk(self):
        if self._use_wrist_offset:
            wrist_pos = np.array(self.data.xpos[self._body_id])
            wrist_rot = np.array(self.data.xmat[self._body_id]).reshape(3, 3)
            return wrist_pos + wrist_rot.dot(self._palm_offset)
        return np.array(self.data.geom_xpos[self._palm_gid])

    def reset(self, seed: int | None=None, options: Dict | None=None):
        super().reset(seed=seed)
        for j in (12, 13, 14):
            self.data.qpos[self.model.jnt_qposadr[j]] = 0.0
            self.data.qvel[self.model.jnt_dofadr[j]] = 0.0
        if self._training_right:
            active_rest = self._right_rest_q
        else:
            active_rest = np.array([25.8, 11.5, 2.9, 28.6, -34.4, 17.2, -14.3], dtype=np.float32) * (math.pi / 180.0)
        noise = self.np_random.uniform(-0.25, 0.25, size=7)
        q0 = active_rest + noise
        for adr, q in zip(self._qadr, q0):
            self.data.qpos[adr] = float(q)
        for qadr, q_des in zip(self._park_qadr, self._park_rest_q):
            self.data.qpos[qadr] = float(q_des)
        for dadr in self._park_dadr:
            self.data.qvel[dadr] = 0.0
        self.data.qvel[:] = 0
        for nm in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'):
            jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid != -1:
                self.data.qpos[self.model.jnt_qposadr[jid]] = 0.0
        self._step_count = 0
        torso_pos = np.array([0.0, 0.0, 0.9])
        self._mujoco.mj_forward(self.model, self.data)
        shoulder_pos = np.array(self.data.xpos[self._shoulder_bid])
        frontal_pref = self.np_random.random() < 0.6
        training_right = self._training_right
        if frontal_pref:
            y_low, y_high = (-0.15, 0.15)
        else:
            side_pref = self.np_random.random() < 0.7
            if side_pref:
                if training_right:
                    y_low, y_high = (-0.45, -0.05)
                else:
                    y_low, y_high = (0.05, 0.45)
            else:
                y_low, y_high = (-0.3, 0.3)
        reach_max = 0.6
        while True:
            if frontal_pref:
                x = 0.1 + self.np_random.random() ** 2 * 0.25
            else:
                x = self.np_random.uniform(0.15, 0.5)
            y = self.np_random.uniform(y_low, y_high)
            horiz = math.hypot(x, y)
            if horiz < 0.35:
                z_hi = 1.3
            else:
                z_hi = 1.1
            z = self.np_random.uniform(0.6, z_hi)
            cand = np.array([x, y, z], dtype=np.float32)
            if np.linalg.norm(cand - torso_pos) <= 0.1:
                continue
            if np.linalg.norm(cand - shoulder_pos) > reach_max:
                continue
            self.p_goal = cand
            if self._goal_mid != -1:
                self.data.mocap_pos[self._goal_mid] = self.p_goal
            break
        self._mujoco.mj_forward(self.model, self.data)
        for j in (12, 13, 14):
            self.data.qpos[self.model.jnt_qposadr[j]] = 0.0
            self.data.qvel[self.model.jnt_dofadr[j]] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        info = {}
        return (obs, info)

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), 'action outside bounds'
        for adr, dq in zip(self._qadr, action):
            self.data.qpos[adr] = np.clip(self.data.qpos[adr] + float(dq), -3.14, 3.14)
        for nm in ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'):
            jid = self._mujoco.mj_name2id(self.model, self._mujoco.mjtObj.mjOBJ_JOINT, nm)
            if jid != -1:
                qadr = int(self.model.jnt_qposadr[jid])
                dadr = int(self.model.jnt_dofadr[jid])
                self.data.qpos[qadr] = 0.0
                self.data.qvel[dadr] = 0.0
        for qadr, q_des in zip(self._park_qadr, self._park_rest_q):
            self.data.qpos[qadr] = float(q_des)
        self.data.qvel[self._park_dadr] = 0.0
        self.data.qvel[self._qadr] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        p_hand = self._fk()
        delta = self.p_goal - p_hand
        dist = np.linalg.norm(delta)
        reward = -5.0 * dist - 0.1 * np.linalg.norm(action)
        ori_mat = np.array(self.data.xmat[self._body_id]).reshape(3, 3)
        grasp_axis_world = ori_mat[:, 0]
        vertical_alignment = abs(float(grasp_axis_world[2]))
        reward -= 1.5 * (1.0 - vertical_alignment)
        palm_x_world = ori_mat[:, 0]
        palm_y_world = ori_mat[:, 1]
        horiz_deviation = abs(float(palm_x_world[2])) + abs(float(palm_y_world[2]))
        reward -= 1.5 * horiz_deviation
        collision_hit = False
        if not self._DISABLE_COLLISION:
            for i in range(self.data.ncon):
                c = self.data.contact[i]
                g1, g2 = (c.geom1, c.geom2)
                if g1 in self._arm_gids and g2 in self._protect_gids or (g2 in self._arm_gids and g1 in self._protect_gids):
                    penetration = max(0.0, -c.dist)
                    if penetration < 0.002:
                        continue
                    reward -= 200.0 + 2000.0 * penetration
                    collision_hit = True
                    break
        if collision_hit:
            g_names = [self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, c.geom1), self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_GEOM, c.geom2)]
            b_names = [self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[c.geom1])), self._mujoco.mj_id2name(self.model, self._mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[c.geom2]))]
            print(f'[env] CONTACT collision @ step {self._step_count}: {g_names[0]}({b_names[0]}) ↔ {g_names[1]}({b_names[1]})')
        torso_pos = np.array([0.0, 0.0, 0.9])
        d_torso = np.linalg.norm(p_hand - torso_pos)
        if d_torso < 0.1:
            reward -= (0.1 - d_torso) * 100.0
        terminated = False
        success = dist < 0.03 and (not collision_hit)
        if success:
            reward += 1.0
            terminated = True
        self._step_count += 1
        if collision_hit and (not self._DISABLE_COLLISION):
            terminated = True
        if self._step_count > self.max_steps:
            terminated = True
        obs = self._get_obs()
        info: Dict[str, float] = {'dist': dist}
        if self._goal_mid != -1:
            self.data.mocap_pos[self._goal_mid] = self.p_goal
        if self.viewer is not None:
            self.viewer.sync()
        return (obs, reward, terminated, False, info)

    def _get_obs(self):
        q = self.data.qpos[self._qadr]
        dq = self.data.qvel[self._qadr]
        p_hand = self._fk()
        delta = self.p_goal - p_hand
        step_frac = np.array([self._step_count / self.max_steps], dtype=np.float32)
        return np.concatenate([q, dq, p_hand, self.p_goal, delta, step_frac]).astype(np.float32)

    def render(self):
        if self.viewer is not None:
            self.viewer.sync()

def make_env(**kwargs):
    return G1ArmReachEnv(**kwargs)
if __name__ == '__main__':
    env = G1ArmReachEnv()
    obs, _ = env.reset()
    done = False
    ep_ret = 0.0
    while not done:
        a = env.action_space.sample()
        obs, r, done, _, _ = env.step(a)
        ep_ret += r
    print('Episode return:', ep_ret)
