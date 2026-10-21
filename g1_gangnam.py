"""Gantry-free MuJoCo playback of the pretrained Gangnam Style G1 policy."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from gangnam_controller import GangnamPolicy
from grit_controller import GRIT_JOINT_NAMES

SCENE = Path(__file__).resolve().parent / "models/g1/scene_29dof.xml"


class GangnamRuntime:
    def __init__(self, policy=None, scene=SCENE):
        self.policy = policy if policy is not None else GangnamPolicy()
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.model.opt.timestep = .002
        self.data = mujoco.MjData(self.model)
        m = self.model
        if m.nq != 36 or m.nv != 35 or m.nu != 29 or m.opt.timestep <= 0:
            raise ValueError("Gangnam requires a free-root 29-joint, 29-motor G1 model")
        self.pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        if self.pelvis < 0 or self.torso < 0:
            raise ValueError("Gangnam requires pelvis and torso_link bodies")
        self.jids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in GRIT_JOINT_NAMES])
        if np.any(self.jids < 0) or len(set(self.jids)) != 29:
            raise ValueError("Gangnam scene is missing named G1 joints")
        self.qids = m.jnt_qposadr[self.jids]
        self.dids = m.jnt_dofadr[self.jids]
        if np.any(m.jnt_type[self.jids] != mujoco.mjtJoint.mjJNT_HINGE) or set(self.qids) != set(range(7, 36)) or set(self.dids) != set(range(6, 35)):
            raise ValueError("Unsupported G1 joint transmission/layout")
        motors = []
        for jid in self.jids:
            matches = np.flatnonzero(m.actuator_trnid[:, 0] == jid)
            if len(matches) != 1:
                raise ValueError("Expected exactly one motor for each named G1 joint")
            aid = int(matches[0])
            if m.actuator_trntype[aid] != mujoco.mjtTrn.mjTRN_JOINT or not np.isclose(m.actuator_gear[aid, 0], 1) or np.any(m.actuator_gear[aid, 1:] != 0) or m.actuator_dyntype[aid] != mujoco.mjtDyn.mjDYN_NONE or m.actuator_gaintype[aid] != mujoco.mjtGain.mjGAIN_FIXED or not np.isclose(m.actuator_gainprm[aid, 0], 1) or m.actuator_biastype[aid] != mujoco.mjtBias.mjBIAS_NONE:
                raise ValueError("Gangnam requires direct unit-gear torque motors")
            motors.append(aid)
        self.motors = np.array(motors)
        if len(set(motors)) != 29 or not np.all(m.jnt_limited[self.jids]) or not np.all(m.actuator_ctrllimited[self.motors]):
            raise ValueError("Gangnam requires bounded joints and torque motors")
        self.limits = m.jnt_range[self.jids]
        self.torque_limits = m.actuator_ctrlrange[self.motors]
        if not np.all(np.isfinite(self.limits)) or not np.all(self.limits[:, 0] < self.limits[:, 1]) or not np.all(np.isfinite(self.torque_limits)) or not np.all(self.torque_limits[:, 0] < self.torque_limits[:, 1]):
            raise ValueError("Invalid joint or torque limits")
        if m.neq != 1 or mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "support_gantry") < 0:
            raise ValueError("Unsupported equality constraints in Gangnam scene")
        if not np.isclose(self.policy.dt / m.opt.timestep, round(self.policy.dt / m.opt.timestep)):
            raise ValueError("Physics timestep must divide the 50Hz policy period")
        self.reset()

    def reset(self):
        self.steps = self.saturated = self.completed_cycles = self.phase_steps = 0
        self.error_sum = 0.
        self.min_z = float("inf")
        self.min_up = float("inf")
        self._restart_excerpt()

    def _restart_excerpt(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.eq_active[:] = 0
        pos, quat, q, dq = self.policy.sample(self.policy.start_time)
        d.qpos[:3], d.qpos[3:7], d.qpos[self.qids] = pos, quat, q
        d.qvel[self.dids] = dq
        h = min(.005, 1 / 60)
        before = self.policy.sample(self.policy.start_time - h)
        after = self.policy.sample(self.policy.start_time + h)
        d.qvel[:3] = (after[0] - before[0]) / (2 * h)
        delta = np.zeros(3)
        mujoco.mju_subQuat(delta, after[1], before[1])
        d.qvel[3:6] = delta / (2 * h)
        mujoco.mj_forward(m, d)
        self.policy.reset(quat, q)
        self.target = np.clip(q.copy(), self.limits[:, 0], self.limits[:, 1])
        self.phase_steps = 0
        self._check_state()

    def _check_state(self):
        d = self.data
        z = float(d.qpos[2])
        up = float(d.xmat[self.pelvis].reshape(3, 3)[2, 2])
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)) or not np.all(np.isfinite(d.xmat)):
            raise RuntimeError("Gangnam simulation became nonfinite")
        self.min_z = min(self.min_z, z)
        self.min_up = min(self.min_up, up)
        if z < .45 or up < .65:
            raise RuntimeError(f"Gangnam fell at {d.time:.3f}s (z={z:.3f}, pelvis_up={up:.3f})")
        if np.any(d.eq_active):
            raise RuntimeError("Gangnam support constraint activated")

    def step_physics(self):
        m, d = self.model, self.data
        if self.phase_steps >= round(self.policy.duration / m.opt.timestep):
            self._check_state()
            self._restart_excerpt()
            print(f"[GANGNAM] Replay cycle {self.completed_cycles + 1}")
        if self.phase_steps % round(self.policy.dt / m.opt.timestep) == 0:
            q = d.qpos[self.qids].copy()
            dq = d.qvel[self.dids].copy()
            omega = d.qvel[3:6].copy()
            torso = d.xmat[self.torso].reshape(3, 3).copy()
            target = np.asarray(self.policy.infer(self.phase_steps * m.opt.timestep, q, dq, omega, torso))
            if target.shape != (29,) or not np.all(np.isfinite(target)):
                raise RuntimeError("Gangnam policy produced invalid targets")
            self.target = np.clip(target, self.limits[:, 0], self.limits[:, 1])
        raw = self.policy.kp * (self.target - d.qpos[self.qids]) - self.policy.kd * d.qvel[self.dids]
        if not np.all(np.isfinite(raw)):
            raise RuntimeError("Gangnam PD torque became nonfinite")
        clipped = np.clip(raw, self.torque_limits[:, 0], self.torque_limits[:, 1])
        self.saturated += int(np.count_nonzero(raw != clipped))
        d.ctrl[self.motors] = clipped
        mujoco.mj_step(m, d)
        self.steps += 1
        self.phase_steps += 1
        self._check_state()
        if self.phase_steps == round(self.policy.duration / m.opt.timestep):
            self.completed_cycles += 1
        ref = self.policy.sample(self.policy.start_time + min(self.phase_steps * m.opt.timestep, self.policy.duration))[2]
        self.error_sum += float(np.sum((d.qpos[self.qids] - ref) ** 2))

    def summary(self):
        return dict(elapsed=self.steps * self.model.opt.timestep, steps=self.steps,
                    completed_cycles=self.completed_cycles,
                    min_z=self.min_z, min_up=self.min_up,
                    torque_saturation=self.saturated / max(1, self.steps * 29),
                    tracking_rms=math.sqrt(self.error_sum / max(1, self.steps * 29)))


def run_gangnam(*, headless=False, duration=None):
    runtime = GangnamRuntime()
    end = runtime.policy.duration if duration is None and headless else (None if duration is None else float(duration))
    if end is not None and (not np.isfinite(end) or end <= 0):
        raise ValueError("Gangnam duration must be positive and finite")
    count = None if end is None else max(1, math.ceil(end / runtime.model.opt.timestep - 1e-10))
    print(f"[GANGNAM] Dedicated pretrained Gangnam Style policy; gantry disabled; scheduled replay {'until close' if end is None else f'{end:.2f}s'}")
    if headless:
        for _ in range(count):
            runtime.step_physics()
    else:
        from gangnam_viewer import run_viewer

        try:
            run_viewer(runtime, count)
        except KeyboardInterrupt:
            print("[GANGNAM] Interrupted by user")
    result = runtime.summary()
    print("[GANGNAM] " + " ".join(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}" for key, value in result.items()))
    return result
