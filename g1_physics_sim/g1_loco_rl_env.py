from __future__ import annotations
import math
import pathlib
from typing import Dict, Tuple, Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

try:
    from .g1_constants import (
        NUM_BODY_MOTORS, JOINT_NAMES_29DOF, NOMINAL_STAND_Q,
        OFFICIAL_KP, OFFICIAL_KD, TORQUE_LIMITS,
        JOINT_LIMITS_MIN, JOINT_LIMITS_MAX, JOINT_LIMIT_MARGIN,
        PHYSICS_DT
    )
except ImportError:
    from g1_constants import (
        NUM_BODY_MOTORS, JOINT_NAMES_29DOF, NOMINAL_STAND_Q,
        OFFICIAL_KP, OFFICIAL_KD, TORQUE_LIMITS,
        JOINT_LIMITS_MIN, JOINT_LIMITS_MAX, JOINT_LIMIT_MARGIN,
        PHYSICS_DT
    )


class G1LocomotionEnv(gym.Env):
    metadata = {'render_modes': ['human', 'none'], 'render_fps': 50}

    def __init__(
        self,
        render_mode: str = 'none',
        target_vx: float = 0.5,
        target_vy: float = 0.0,
        target_wz: float = 0.0,
        xml_path: Optional[str] = None,
        max_episode_steps: int = 500,
        decimation: int = 10,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.target_vx = float(target_vx)
        self.target_vy = float(target_vy)
        self.target_wz = float(target_wz)
        self.max_episode_steps = max_episode_steps
        self.decimation = decimation
        self.dt = PHYSICS_DT * decimation
        self.num_motors = NUM_BODY_MOTORS

        base = pathlib.Path(__file__).resolve().parent
        if xml_path is None:
            xml_path = str(base / 'models' / 'g1' / 'g1_physics_scene.xml')

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = PHYSICS_DT

        self.pelvis_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')

        self._qpos_adr = np.zeros(self.num_motors, dtype=np.int32)
        self._dof_adr = np.zeros(self.num_motors, dtype=np.int32)
        self._actuator_id = np.zeros(self.num_motors, dtype=np.int32)

        for i, name in enumerate(JOINT_NAMES_29DOF):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                raise RuntimeError(f"Joint '{name}' not found in {xml_path}")
            self._qpos_adr[i] = int(self.model.jnt_qposadr[jid])
            self._dof_adr[i] = int(self.model.jnt_dofadr[jid])
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name.replace('_joint', ''))
            self._actuator_id[i] = aid

        self.action_scale = np.full(self.num_motors, 0.35, dtype=np.float32)
        self.action_scale[15:] = 0.25

        self.kp = OFFICIAL_KP.copy()
        self.kd = OFFICIAL_KD.copy()
        self.nominal_q = NOMINAL_STAND_Q[:self.num_motors].copy()

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_motors,), dtype=np.float32
        )

        obs_dim = 1 + 3 + 3 + 3 + self.num_motors + self.num_motors + self.num_motors + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self.prev_action = np.zeros(self.num_motors, dtype=np.float32)
        self.step_count = 0
        self.viewer = None

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = 0.793
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        noise = self.np_random.uniform(-0.02, 0.02, size=self.num_motors).astype(np.float32)
        init_q = self.nominal_q + noise

        for i in range(self.num_motors):
            self.data.qpos[self._qpos_adr[i]] = float(init_q[i])

        self.data.qvel[:] = 0.0
        self.prev_action[:] = 0.0
        self.step_count = 0

        mujoco.mj_forward(self.model, self.data)
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self) -> np.ndarray:
        z = float(self.data.qpos[2])

        R = self.data.xmat[self.pelvis_bid].reshape(3, 3)
        gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        projected_gravity = R.T @ gravity_world

        world_linvel = self.data.qvel[0:3]
        body_linvel = R.T @ world_linvel

        body_angvel = self.data.qvel[3:6]

        joint_pos = np.array([self.data.qpos[adr] for adr in self._qpos_adr], dtype=np.float32)
        joint_vel = np.array([self.data.qvel[adr] for adr in self._dof_adr], dtype=np.float32)

        q_err = joint_pos - self.nominal_q
        cmd = np.array([self.target_vx, self.target_vy, self.target_wz], dtype=np.float32)

        obs = np.concatenate([
            [z],
            projected_gravity.astype(np.float32),
            body_linvel.astype(np.float32),
            body_angvel.astype(np.float32),
            q_err,
            joint_vel,
            self.prev_action,
            cmd
        ]).astype(np.float32)

        return np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        q_target = self.nominal_q + action * self.action_scale
        q_target = np.clip(
            q_target,
            JOINT_LIMITS_MIN[:self.num_motors] + JOINT_LIMIT_MARGIN,
            JOINT_LIMITS_MAX[:self.num_motors] - JOINT_LIMIT_MARGIN
        )

        for _ in range(self.decimation):
            cur_q = np.array([self.data.qpos[adr] for adr in self._qpos_adr], dtype=np.float32)
            cur_dq = np.array([self.data.qvel[adr] for adr in self._dof_adr], dtype=np.float32)
            tau = self.kp * (q_target - cur_q) - self.kd * cur_dq
            tau = np.clip(tau, -TORQUE_LIMITS[:self.num_motors], TORQUE_LIMITS[:self.num_motors])

            for i in range(self.num_motors):
                aid = self._actuator_id[i]
                if aid != -1:
                    self.data.ctrl[aid] = tau[i]

            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._get_obs()

        terminated, term_reason = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps

        reward = self._compute_reward(action, terminated)
        self.prev_action = action.copy()

        info = {
            'step': self.step_count,
            'reason': term_reason,
            'vx': float(self.data.qvel[0]),
            'z': float(self.data.qpos[2])
        }

        if self.render_mode == 'human':
            self.render()

        return obs, reward, terminated, truncated, info

    def _is_terminated(self) -> Tuple[bool, str]:
        z = float(self.data.qpos[2])
        if z < 0.48:
            return True, 'fallen_too_low'
        if z > 1.05:
            return True, 'height_too_high'

        R = self.data.xmat[self.pelvis_bid].reshape(3, 3)
        up_z = float(R[2, 2])
        if up_z < 0.707:
            return True, 'tilted_too_much'

        if not np.all(np.isfinite(self.data.qpos[:7])):
            return True, 'nan_qpos'
        if not np.all(np.isfinite(self.data.qvel[:6])):
            return True, 'nan_qvel'

        return False, ''

    def _compute_reward(self, action: np.ndarray, terminated: bool) -> float:
        if terminated:
            return -10.0

        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        wz = float(self.data.qvel[5])
        z = float(self.data.qpos[2])

        R = self.data.xmat[self.pelvis_bid].reshape(3, 3)
        upright = float(max(0.0, R[2, 2]))

        r_vx = 3.5 * math.exp(-((vx - self.target_vx) ** 2) / 0.15)
        r_vy = -1.5 * (vy ** 2)
        r_wz = 1.0 * math.exp(-((wz - self.target_wz) ** 2) / 0.25)
        r_height = 2.0 * math.exp(-((z - 0.77) ** 2) / 0.02)
        r_upright = 2.5 * upright

        act_diff = action - self.prev_action
        r_smooth = -0.01 * float(np.sum(act_diff ** 2))
        r_alive = 2.0

        total_reward = r_vx + r_vy + r_wz + r_height + r_upright + r_smooth + r_alive
        return float(total_reward)

    def render(self):
        if self.viewer is None:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
