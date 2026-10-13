"""
Reference policies for the Unitree G1.

AnkleSwingPolicy reproduces the official
official_unitree_repos/unitree_sdk2/example/g1/low_level/g1_ankle_swing_example.cpp
including its PR/AB ankle mode switching and the 30 deg / 10 deg amplitudes.

SAFETY: the official example ramps the legs to the ZERO posture. On the real
robot the zero posture is not a stable standing pose, so this policy is only
valid while the robot is suspended from a gantry. ArmSwingPolicy is the safer
first test because it leaves the legs at NOMINAL_STAND_Q.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

try:  # package and flat-script imports both work
    from .g1_constants import (    NUM_BODY_MOTORS, NOMINAL_STAND_Q, ZERO_POSE_Q,
    OFFICIAL_KP, OFFICIAL_KD, ARM_SDK_KP, ARM_SDK_KD,
    MODE_PR, MODE_AB, MOTOR_MODE_ENABLE, MOTOR_MODE_DISABLE,
    LEG_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, WAIST_JOINTS,
    L_HIP_PITCH, L_HIP_ROLL, L_HIP_YAW, L_KNEE, L_ANKLE_PITCH, L_ANKLE_ROLL,
    R_HIP_PITCH, R_HIP_ROLL, R_HIP_YAW, R_KNEE, R_ANKLE_PITCH, R_ANKLE_ROLL,
    L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW, L_ELBOW, L_WRIST_ROLL,
    R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW, R_ELBOW, R_WRIST_ROLL,
    ARM_REST_17, ANKLE_SWING_P_MAX, ANKLE_SWING_R_MAX, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (NUM_BODY_MOTORS, NOMINAL_STAND_Q, ZERO_POSE_Q,
OFFICIAL_KP, OFFICIAL_KD, ARM_SDK_KP, ARM_SDK_KD,
MODE_PR, MODE_AB, MOTOR_MODE_ENABLE, MOTOR_MODE_DISABLE,
LEG_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, WAIST_JOINTS,
L_HIP_PITCH, L_HIP_ROLL, L_HIP_YAW, L_KNEE, L_ANKLE_PITCH, L_ANKLE_ROLL,
R_HIP_PITCH, R_HIP_ROLL, R_HIP_YAW, R_KNEE, R_ANKLE_PITCH, R_ANKLE_ROLL,
L_SHOULDER_PITCH, L_SHOULDER_ROLL, L_SHOULDER_YAW, L_ELBOW, L_WRIST_ROLL,
R_SHOULDER_PITCH, R_SHOULDER_ROLL, R_SHOULDER_YAW, R_ELBOW, R_WRIST_ROLL,
ARM_REST_17, ANKLE_SWING_P_MAX, ANKLE_SWING_R_MAX, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX)
try:  # package and flat-script imports both work
    from .g1_state import (    G1Action,
    G1Observation,
    G1State)
except ImportError:  # pragma: no cover - flat script layout
    from g1_state import (G1Action,
G1Observation,
G1State)

LO = JOINT_LIMITS_MIN[:NUM_BODY_MOTORS] + 0.03
HI = JOINT_LIMITS_MAX[:NUM_BODY_MOTORS] - 0.03


def _clip(q: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(q, dtype=np.float32), LO, HI)


class _BasePolicy:
    def configure(self, config: Dict[str, Any]) -> None:
        pass

    def get_info(self) -> Dict[str, Any]:
        return {}

    def _obs(self, state: G1State) -> G1Observation:
        return G1Observation(
            joint_pos=state.q,
            joint_vel=state.dq,
            base_linvel=state.base_linvel,
            base_angvel=state.base_angvel,
            left_foot_contact=state.left_foot_contact,
            right_foot_contact=state.right_foot_contact,
            timestamp=state.timestamp,
        )


class StandPolicy(_BasePolicy):
    """Hold NOMINAL_STAND_Q at the official gains. Safest first real-robot test."""

    def __init__(self, ramp_time: float = 2.0, kp: Optional[np.ndarray] = None,
                 kd: Optional[np.ndarray] = None, engage_motors: bool = True):
        self.ramp_time = max(0.0, ramp_time)
        self.kp = OFFICIAL_KP.copy() if kp is None else np.asarray(kp, np.float32)
        self.kd = OFFICIAL_KD.copy() if kd is None else np.asarray(kd, np.float32)
        self.engage_motors = engage_motors
        self._t0: Optional[float] = None

    def reset(self, state: G1State) -> None:
        self._t0 = state.timestamp

    def step(self, state: G1State) -> G1Action:
        a = G1Action(kp=self.kp, kd=self.kd)
        if self._t0 is None:
            self._t0 = state.timestamp
        r = 1.0
        if self.ramp_time > 0:
            r = float(np.clip((state.timestamp - self._t0) / self.ramp_time, 0.0, 1.0))
        a.q_des[:] = _clip((1.0 - r) * state.q + r * NOMINAL_STAND_Q)
        a.motor_mode[:] = MOTOR_MODE_ENABLE if self.engage_motors else MOTOR_MODE_DISABLE
        return a

    def get_observation(self, state: G1State) -> G1Observation:
        return self._obs(state)

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return NUM_BODY_MOTORS * 2 + 9


class AnkleSwingPolicy(_BasePolicy):
    """
    Faithful port of the official ankle-swing demo.

    Stage 1 (0..T)        ramp every joint from measured q to the zero posture
    Stage 2 (T..2T)       mode_pr = PR, ankle pitch/roll sine sweep
    Stage 3 (2T..)        mode_pr = AB, ankle A/B sine sweep

    REQUIRES A GANTRY. The zero posture is not balanceable.
    """

    def __init__(self, duration: float = 3.0, control_dt: float = 0.002):
        self.duration = float(duration)
        self.control_dt = float(control_dt)
        self.time = 0.0
        self.done = False
        self._started = False

    def reset(self, state: G1State) -> None:
        self.time = 0.0
        self.done = False
        self._started = True

    def step(self, state: G1State) -> G1Action:
        if not self._started:
            self.reset(state)
        T = self.duration
        self.time += self.control_dt
        t = self.time

        a = G1Action()
        a.q_des[:] = 0.0
        a.kp[:] = OFFICIAL_KP
        a.kd[:] = OFFICIAL_KD

        if t < T:
            # Stage 1: blend measured -> zero posture, do not jump.
            ratio = float(np.clip(t / T, 0.0, 1.0))
            a.q_des[:] = (1.0 - ratio) * state.q
            a.mode_pr = MODE_PR
        elif t < T * 2:
            # Stage 2: PR mode - series pitch/roll control.
            a.mode_pr = MODE_PR
            s = t - T
            p = ANKLE_SWING_P_MAX * math.sin(2.0 * math.pi * s)
            r = ANKLE_SWING_R_MAX * math.sin(2.0 * math.pi * s)
            a.q_des[L_ANKLE_PITCH] = p
            a.q_des[L_ANKLE_ROLL] = r
            a.q_des[R_ANKLE_PITCH] = p
            a.q_des[R_ANKLE_ROLL] = -r
        else:
            # Stage 3: AB mode - parallel A/B control.
            a.mode_pr = MODE_AB
            s = t - T * 2
            a_big = ANKLE_SWING_P_MAX * math.sin(math.pi * s)
            a_small = ANKLE_SWING_R_MAX * math.sin(math.pi * s + math.pi)
            a.q_des[L_ANKLE_PITCH] = a_big      # LeftAnkleB
            a.q_des[L_ANKLE_ROLL] = a_small     # LeftAnkleA
            a.q_des[R_ANKLE_PITCH] = -a_big     # RightAnkleB
            a.q_des[R_ANKLE_ROLL] = -a_small    # RightAnkleA
            if t > T * 4:
                self.done = True

        a.q_des[:] = _clip(a.q_des)
        a.motor_mode[:] = MOTOR_MODE_ENABLE
        return a

    def get_observation(self, state: G1State) -> G1Observation:
        return self._obs(state)

    def get_info(self) -> Dict[str, Any]:
        return {'time': self.time, 'done': self.done}

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return NUM_BODY_MOTORS * 2 + 9


class ArmSwingPolicy(_BasePolicy):
    """
    Arm_sdk-safe demo: legs stay at NOMINAL_STAND_Q, only the 17 upper-body
    joints move. Uses the official arm_sdk gains and the official arm7 targets.
    """

    def __init__(self, duration: float = 5.0, right_arm: bool = False,
                 kp: float = ARM_SDK_KP, kd: float = ARM_SDK_KD, control_dt: float = 0.02):
        self.duration = float(duration)
        self.kp = kp
        self.kd = kd
        self.control_dt = control_dt
        self.arm = RIGHT_ARM_JOINTS if right_arm else LEFT_ARM_JOINTS
        self.idle = LEFT_ARM_JOINTS if right_arm else RIGHT_ARM_JOINTS
        self.t = 0.0

    def _targets(self) -> np.ndarray:
        """Map ARM_REST_17 onto the selected arm plus waist."""
        q = NOMINAL_STAND_Q.copy()
        if self.arm is RIGHT_ARM_JOINTS:
            rest = ARM_REST_17[7:14]
        else:
            rest = ARM_REST_17[0:7]
        q[self.arm] = rest
        q[WAIST_JOINTS] = ARM_REST_17[14:17]
        return _clip(q)

    def reset(self, state: G1State) -> None:
        self.t = 0.0

    def step(self, state: G1State) -> G1Action:
        self.t += self.control_dt
        target = self._targets()

        a = G1Action()
        a.kp[:] = self.kp
        a.kd[:] = self.kd
        upper = WAIST_JOINTS + self.arm

        # Slew the upper body at the official 0.5 rad/s; hold legs steady.
        max_delta = 0.5 * self.control_dt
        cur = state.q.copy()
        nxt = np.clip(target, cur - max_delta, cur + max_delta)
        a.q_des[:] = nxt
        a.q_des[LEG_JOINTS] = NOMINAL_STAND_Q[LEG_JOINTS]
        a.q_des[self.idle] = NOMINAL_STAND_Q[self.idle]
        a.q_des[:] = _clip(a.q_des)

        a.motor_mode[:] = MOTOR_MODE_ENABLE
        a.arm_sdk_weight = 1.0
        return a

    def get_observation(self, state: G1State) -> G1Observation:
        return self._obs(state)

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return NUM_BODY_MOTORS * 2 + 9


class WavePolicy(_BasePolicy):
    """Friendly wave. Legs held at NOMINAL_STAND_Q so it is gantry-optional."""

    def __init__(self, right_arm: bool = False, duration: float = 4.0,
                 kp: float = 40.0, kd: float = 1.0, control_dt: float = 0.02):
        self.right_arm = right_arm
        self.duration = float(duration)
        self.kp = kp
        self.kd = kd
        self.control_dt = control_dt
        self.t = 0.0
        b = -1.0 if right_arm else 1.0
        self.side = b

    def reset(self, state: G1State) -> None:
        self.t = 0.0

    def step(self, state: G1State) -> G1Action:
        self.t += self.control_dt
        t = self.t
        a = G1Action()
        a.q_des[:] = NOMINAL_STAND_Q
        a.kp[:] = self.kp
        a.kd[:] = self.kd

        if self.right_arm:
            base = R_SHOULDER_PITCH
            idle = LEFT_ARM_JOINTS
        else:
            base = L_SHOULDER_PITCH
            idle = RIGHT_ARM_JOINTS

        a.q_des[base] = -0.6
        a.q_des[base + 1] = self.side * 0.7
        a.q_des[base + 3] = 1.2
        a.q_des[base + 4] = 0.45 * math.sin(2.0 * math.pi * 2.5 * t)
        a.q_des[base + 2] = self.side * (0.2 + 0.15 * math.cos(2.0 * math.pi * 2.5 * t))
        a.q_des[idle] = NOMINAL_STAND_Q[idle]
        a.q_des[:] = _clip(a.q_des)
        a.motor_mode[:] = MOTOR_MODE_ENABLE
        return a

    def get_observation(self, state: G1State) -> G1Observation:
        return self._obs(state)

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return NUM_BODY_MOTORS * 2 + 9


class JointPositionPolicy(_BasePolicy):
    """Move to an explicit 29-vector over a ramp time."""

    def __init__(self, target_q: np.ndarray, ramp_time: float = 2.0,
                 kp: Optional[np.ndarray] = None, kd: Optional[np.ndarray] = None):
        self.target_q = _clip(np.asarray(target_q, dtype=np.float32).reshape(NUM_BODY_MOTORS))
        self.ramp_time = max(1e-3, ramp_time)
        self.kp = OFFICIAL_KP.copy() if kp is None else np.asarray(kp, np.float32)
        self.kd = OFFICIAL_KD.copy() if kd is None else np.asarray(kd, np.float32)
        self._t0: Optional[float] = None

    def reset(self, state: G1State) -> None:
        self._t0 = state.timestamp

    def step(self, state: G1State) -> G1Action:
        if self._t0 is None:
            self._t0 = state.timestamp
        r = float(np.clip((state.timestamp - self._t0) / self.ramp_time, 0.0, 1.0))
        a = G1Action(kp=self.kp, kd=self.kd)
        a.q_des[:] = _clip((1.0 - r) * state.q + r * self.target_q)
        a.motor_mode[:] = MOTOR_MODE_ENABLE
        return a

    def get_observation(self, state: G1State) -> G1Observation:
        return self._obs(state)

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return NUM_BODY_MOTORS * 2 + 9


class LocomotionPolicy(_BasePolicy):
    """
    ONNX locomotion policy wrapper.

    Observations use the standard layout. `base_linvel` is deliberately NOT fed to
    the actor because it is not measurable from robot proprioception alone; if the
    exported policy expects it, pass it explicitly and expect degraded sim2real.
    """

    def __init__(self, policy_path: Optional[str] = None, command: Optional[np.ndarray] = None):
        self.command = np.zeros(3, dtype=np.float32) if command is None \
            else np.asarray(command, dtype=np.float32).reshape(3)
        self._sess = None
        self._last_action = np.zeros(len(LEG_JOINTS), dtype=np.float32)
        if policy_path:
            self.load(policy_path)

    def load(self, policy_path: str) -> None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(policy_path, opts, providers=['CPUExecutionProvider'])
        self._input_name = self._sess.get_inputs()[0].name

    def reset(self, state: G1State) -> None:
        self._last_action[:] = 0.0

    def _obs(self, state: G1State) -> np.ndarray:  # type: ignore[override]
        gravity = state.imu_linacc / max(1e-6, np.linalg.norm(state.imu_linacc))
        return np.concatenate([
            state.q[LEG_JOINTS],
            state.dq[LEG_JOINTS],
            state.imu_angvel,
            gravity,
            np.array([float(state.left_foot_contact), float(state.right_foot_contact)], np.float32),
            self.command,
            self._last_action,
        ]).astype(np.float32)

    def step(self, state: G1State) -> G1Action:
        a = G1Action()
        a.q_des[:] = NOMINAL_STAND_Q
        a.kp[:] = OFFICIAL_KP
        a.kd[:] = OFFICIAL_KD

        if self._sess is not None:
            try:
                out = self._sess.run(None, {self._input_name: self._obs(state)})[0].reshape(-1)
                n = min(out.shape[0], len(LEG_JOINTS))
                self._last_action[:n] = out[:n]
            except Exception as e:
                print(f'[LocomotionPolicy] inference failed: {e}; holding stand pose')
                self._sess = None

        if self._sess is not None:
            a.q_des[LEG_JOINTS] = NOMINAL_STAND_Q[LEG_JOINTS] + self._last_action

        # Legs are enabled here because this policy publishes to rt/lowcmd.
        a.motor_mode[:] = MOTOR_MODE_ENABLE
        a.q_des[:] = _clip(a.q_des)
        return a

    def get_observation(self, state: G1State) -> G1Observation:
        return G1Observation(
            joint_pos=state.q, joint_vel=state.dq,
            base_linvel=state.base_linvel, base_angvel=state.base_angvel,
            imu_angvel=state.imu_angvel, imu_linacc=state.imu_linacc,
            left_foot_contact=state.left_foot_contact,
            right_foot_contact=state.right_foot_contact,
            command=self.command, actions=self._last_action,
            timestamp=state.timestamp,
        )

    @property
    def action_dim(self) -> int:
        return NUM_BODY_MOTORS

    @property
    def obs_dim(self) -> int:
        return NUM_BODY_MOTORS * 2 + 9


# Backwards-compatible aliases for the old names used by run_sim.py / run_simulation.py
AnkleSwingPolicy.__doc__ = AnkleSwingPolicy.__doc__
WavePolicy = WavePolicy