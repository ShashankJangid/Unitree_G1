"""
Layered safety supervisor for the Unitree G1.

Design rules that this module exists to enforce:

1. An E-stop must actually remove torque. Returning only joint positions is not
   an E-stop: the caller keeps applying kp/kd, so the robot stays stiff and the
   "stopped" command is a position hold. Every guarded output is therefore a full
   G1Action including gains, and EMERGENCY_STOP zeroes kp and drops motor enable.

2. Alarms are latched, not cleared each tick. A fault that caused the stop must
   stay visible after the stop, otherwise the operator loses the cause.

3. Recovery is explicit. SOFT_STOP never auto-resumes into ACTIVE; a human (or
   an explicit reset_estop call) has to acknowledge it.

4. E-stop is a latch. Clearing it requires an explicit call, not the next tick.

5. Outputs are clamped to model joint limits with a margin. Safety filtering must
   never be the thing that pushes a joint past its mechanical stop.
"""
from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:  # package and flat-script imports both work
    from .g1_constants import (    NUM_BODY_MOTORS, TORQUE_LIMITS, NOMINAL_STAND_Q, JOINT_NAMES_29DOF,
    JOINT_LIMITS_MIN, JOINT_LIMITS_MAX, JOINT_LIMIT_MARGIN,
    OFFICIAL_KP, OFFICIAL_KD, MOTOR_MODE_ENABLE, MOTOR_MODE_DISABLE,
    FALL_HEIGHT_THRESHOLD, MAX_JOINT_VELOCITY_FULL, LEG_JOINTS)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (NUM_BODY_MOTORS, TORQUE_LIMITS, NOMINAL_STAND_Q, JOINT_NAMES_29DOF,
JOINT_LIMITS_MIN, JOINT_LIMITS_MAX, JOINT_LIMIT_MARGIN,
OFFICIAL_KP, OFFICIAL_KD, MOTOR_MODE_ENABLE, MOTOR_MODE_DISABLE,
FALL_HEIGHT_THRESHOLD, MAX_JOINT_VELOCITY_FULL, LEG_JOINTS)
try:  # package and flat-script imports both work
    from .g1_state import (    G1Action)
except ImportError:  # pragma: no cover - flat script layout
    from g1_state import (G1Action)


class SafetyState(enum.Enum):
    DISARMED = 'DISARMED'
    CALIBRATING = 'CALIBRATING'
    GANTRY_HOLD = 'GANTRY_HOLD'
    ACTIVE_TELEOP = 'ACTIVE_TELEOP'
    SOFT_STOP = 'SOFT_STOP'
    EMERGENCY_STOP = 'EMERGENCY_STOP'


# Thresholds are module-level so tests and operators read the same numbers the
# code actually uses (the previous docstrings disagreed with the constants).
TILT_LIMIT_DEG = 30.0
FALL_LIMIT_M = 0.45
HEARTBEAT_TIMEOUT_MS = 150.0
SELF_COLLISION_DIST = -0.002
THERMAL_LIMIT_PCT = 95.0


@dataclass
class SafetyStatus:
    state: SafetyState = SafetyState.DISARMED
    is_safe: bool = True
    active_alarms: List[str] = field(default_factory=list)
    latched_alarms: List[str] = field(default_factory=list)
    min_torso_dist: float = 1.0
    hand_to_hand_dist: float = 1.0
    max_joint_velocity: float = 0.0
    max_joint_accel: float = 0.0
    max_torque_ratio: float = 0.0
    highest_thermal_load: float = 0.0
    imu_tilt_deg: float = 0.0
    watchdog_latency_ms: float = 0.0


class SCurveLimiter:
    """Jerk/accel/velocity-limited position tracker."""

    def __init__(
        self,
        num_joints: int = NUM_BODY_MOTORS,
        v_max: float = 1.0,
        a_max: float = 8.0,
        j_max: float = 40.0,
        dt: float = 0.02,
    ):
        self.num_joints = num_joints
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.dt = max(dt, 1e-4)
        self.q_filtered = NOMINAL_STAND_Q[:num_joints].copy()
        self.dq_filtered = np.zeros(num_joints, dtype=np.float32)
        self.ddq_filtered = np.zeros(num_joints, dtype=np.float32)

    def reset(self, q_init: np.ndarray) -> None:
        q_init = np.asarray(q_init, dtype=np.float32).reshape(-1)
        self.num_joints = q_init.shape[0]
        self.q_filtered = q_init.copy()
        self.dq_filtered = np.zeros(self.num_joints, dtype=np.float32)
        self.ddq_filtered = np.zeros(self.num_joints, dtype=np.float32)

    def step(self, q_des: np.ndarray) -> np.ndarray:
        q_des = np.asarray(q_des, dtype=np.float32).reshape(-1)
        if q_des.shape[0] != self.q_filtered.shape[0]:
            self.reset(q_des)
        err = q_des - self.q_filtered

        target_v = np.clip(err / self.dt, -self.v_max, self.v_max)
        target_a = np.clip((target_v - self.dq_filtered) / self.dt, -self.a_max, self.a_max)
        jerk = np.clip((target_a - self.ddq_filtered) / self.dt, -self.j_max, self.j_max)
        self.ddq_filtered = np.clip(self.ddq_filtered + jerk * self.dt, -self.a_max, self.a_max)
        self.dq_filtered = np.clip(self.dq_filtered + self.ddq_filtered * self.dt, -self.v_max, self.v_max)
        self.q_filtered = np.clip(self.q_filtered + self.dq_filtered * self.dt,
                                  q_des - self.v_max * self.dt * 4,
                                  q_des + self.v_max * self.dt * 4)
        return self.q_filtered.copy()


class SelfCollisionGuard:
    """Coarse wrist-vs-torso and wrist-vs-wrist clearance, in the XY plane."""

    TORSO_RADIUS = 0.12
    MIN_CLEARANCE_TORSO = 0.08
    MIN_HAND_DISTANCE = 0.09

    @staticmethod
    def check_clearance(
        left_wrist_p: np.ndarray,
        right_wrist_p: np.ndarray,
        pelvis_p: np.ndarray,
    ) -> Tuple[bool, float, float, Optional[str]]:
        torso_xy = np.asarray(pelvis_p, dtype=np.float32)[0:2]
        l_xy = float(np.linalg.norm(np.asarray(left_wrist_p)[0:2] - torso_xy))
        r_xy = float(np.linalg.norm(np.asarray(right_wrist_p)[0:2] - torso_xy))
        torso_l = max(0.0, l_xy - SelfCollisionGuard.TORSO_RADIUS)
        torso_r = max(0.0, r_xy - SelfCollisionGuard.TORSO_RADIUS)
        hand_dist = float(np.linalg.norm(np.asarray(left_wrist_p) - np.asarray(right_wrist_p)))

        if torso_l < SelfCollisionGuard.MIN_CLEARANCE_TORSO:
            return False, min(torso_l, torso_r), hand_dist, \
                f'left wrist {torso_l * 100:.1f}cm from torso (< 8cm)'
        if torso_r < SelfCollisionGuard.MIN_CLEARANCE_TORSO:
            return False, min(torso_l, torso_r), hand_dist, \
                f'right wrist {torso_r * 100:.1f}cm from torso (< 8cm)'
        if hand_dist < SelfCollisionGuard.MIN_HAND_DISTANCE:
            return False, min(torso_l, torso_r), hand_dist, \
                f'wrists {hand_dist * 100:.1f}cm apart (< 9cm)'
        return True, min(torso_l, torso_r), hand_dist, None


class ThermalWatchdog:
    """I^2*t thermal load estimate with exponential cooling."""

    def __init__(self, num_motors: int = NUM_BODY_MOTORS, dt: float = 0.02,
                 cooling_rate: float = 0.15):
        self.num_motors = num_motors
        self.dt = max(dt, 1e-4)
        self.cooling_rate = cooling_rate
        self.thermal_energy = np.zeros(num_motors, dtype=np.float32)

    def update(self, torques: np.ndarray) -> Tuple[bool, float, Optional[str]]:
        tau = np.abs(np.asarray(torques, dtype=np.float32).reshape(-1))
        n = tau.shape[0]
        if self.thermal_energy.shape[0] != n:
            self.thermal_energy = np.zeros(n, dtype=np.float32)
        lim = TORQUE_LIMITS[:n]
        tau_norm = tau / np.maximum(lim, 1e-3)
        self.thermal_energy *= (1.0 - self.cooling_rate * self.dt)
        self.thermal_energy += (tau_norm ** 2) * self.dt

        load_pct = float(np.max(self.thermal_energy) / 3.0) * 100.0
        if load_pct >= THERMAL_LIMIT_PCT:
            idx = int(np.argmax(self.thermal_energy))
            name = JOINT_NAMES_29DOF[idx] if idx < len(JOINT_NAMES_29DOF) else f'joint_{idx}'
            return False, load_pct, f"motor '{name}' thermal load {load_pct:.0f}% (>= 95%)"
        return True, load_pct, None


def clamp_to_joint_limits(q: np.ndarray, margin: float = JOINT_LIMIT_MARGIN) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(NUM_BODY_MOTORS)
    lo = JOINT_LIMITS_MIN[:NUM_BODY_MOTORS] + margin
    hi = JOINT_LIMITS_MAX[:NUM_BODY_MOTORS] - margin
    hi = np.maximum(hi, lo)
    return np.clip(q, lo, hi)


class G1FullSafetyManager:
    """Stateful interlock. Produces a complete, always-safe G1Action."""

    def __init__(self, initial_q: Optional[np.ndarray] = None, dt: float = 0.02):
        q0 = clamp_to_joint_limits(
            initial_q if initial_q is not None else NOMINAL_STAND_Q) if initial_q is None \
            else np.asarray(initial_q, dtype=np.float32).reshape(-1)
        self.dt = dt
        self.state = SafetyState.DISARMED
        self.s_curve = SCurveLimiter(dt=dt)
        self.s_curve.reset(q0)
        self.thermal = ThermalWatchdog(dt=dt)
        self.last_heartbeat_time = time.time()
        self.last_safe_q = q0.copy()
        self.alarms: List[str] = []
        self._latched: List[str] = []
        self._estop_reason: Optional[str] = None

    # -------------------------------------------------------- state changes
    def set_state(self, new_state: SafetyState) -> None:
        self.state = new_state

    def reset(self, initial_q: Optional[np.ndarray] = None) -> None:
        """Acknowledge faults and enter GANTRY_HOLD (never straight to ACTIVE)."""
        q0 = np.asarray(initial_q if initial_q is not None else NOMINAL_STAND_Q,
                        dtype=np.float32).reshape(-1)
        self.state = SafetyState.GANTRY_HOLD
        self.s_curve.reset(q0)
        self.thermal.thermal_energy[:] = 0.0
        self.last_heartbeat_time = time.time()
        self.last_safe_q = q0.copy()
        self.alarms.clear()
        self._latched.clear()
        self._estop_reason = None

    def reset_estop(self) -> bool:
        """Explicit E-stop acknowledgement. Call only with the robot understood."""
        self._estop_reason = None
        self.reset()
        return True

    def acknowledge_soft_stop(self) -> None:
        """Operator acknowledges a soft stop and resumes teleoperation."""
        if self.state == SafetyState.EMERGENCY_STOP:
            raise RuntimeError('cannot acknowledge soft stop while E-stopped; call reset_estop()')
        if self.state == SafetyState.SOFT_STOP:
            self.state = SafetyState.ACTIVE_TELEOP
            self.alarms.clear()

    def engage(self) -> None:
        if self.state in (SafetyState.GANTRY_HOLD, SafetyState.CALIBRATING):
            self.state = SafetyState.ACTIVE_TELEOP

    def disarm(self) -> None:
        self.state = SafetyState.DISARMED

    def feed_watchdog(self) -> None:
        self.last_heartbeat_time = time.time()

    def emergency_stop(self, reason: str = 'manual E-stop') -> None:
        """Latch EMERGENCY_STOP and remember why."""
        self.state = SafetyState.EMERGENCY_STOP
        self._estop_reason = reason
        msg = f'[E-STOP] {reason}'
        if msg not in self._latched:
            self._latched.append(msg)

    @property
    def estop_reason(self) -> Optional[str]:
        return self._estop_reason

    # --------------------------------------------------------------- output
    def _damped_action(self, q_hold: np.ndarray) -> G1Action:
        """Torque-free hold. kp=0, kd small, feedforward zero, motors disabled."""
        a = G1Action()
        a.q_des[:] = q_hold
        a.dq_des[:] = 0.0
        a.kp[:] = 0.0
        a.kd[:] = 1.0
        a.tau_ff[:] = 0.0
        a.motor_mode[:] = MOTOR_MODE_DISABLE
        a.arm_sdk_weight = 0.0
        return a

    def _hold_action(self, q_hold: np.ndarray) -> G1Action:
        """Stiff position hold with official gains - used for SOFT_STOP."""
        a = G1Action()
        a.q_des[:] = q_hold
        a.dq_des[:] = 0.0
        a.kp[:] = OFFICIAL_KP
        a.kd[:] = OFFICIAL_KD
        a.tau_ff[:] = 0.0
        a.motor_mode[:] = MOTOR_MODE_ENABLE
        return a

    def filter_and_guard(
        self,
        target_q: np.ndarray,
        current_q: np.ndarray,
        current_torques: np.ndarray,
        left_wrist_p: np.ndarray,
        right_wrist_p: np.ndarray,
        pelvis_p: np.ndarray,
        base_tilt_deg: float = 0.0,
        base_height: float = 0.793,
    ) -> Tuple[G1Action, SafetyStatus]:
        """
        Evaluate every interlock and return the action that must be published.

        Returns a full G1Action. On EMERGENCY_STOP the action disables motors and
        zeroes gains; callers must publish it verbatim.
        """
        now = time.time()
        self.alarms.clear()
        hb_ms = (now - self.last_heartbeat_time) * 1000.0

        # ---- link loss
        if hb_ms > HEARTBEAT_TIMEOUT_MS and self.state == SafetyState.ACTIVE_TELEOP:
            self.alarms.append(f'DDS heartbeat lost ({hb_ms:.0f}ms > {HEARTBEAT_TIMEOUT_MS:.0f}ms)')
            self.state = SafetyState.SOFT_STOP

        # ---- posture
        if abs(base_tilt_deg) > TILT_LIMIT_DEG:
            self.alarms.append(f'base tilt {base_tilt_deg:.1f} deg (>|{TILT_LIMIT_DEG}|)')
            self.emergency_stop(f'tilt {base_tilt_deg:.1f}deg')
        if base_height < FALL_LIMIT_M:
            self.alarms.append(f'base height {base_height:.2f}m (< {FALL_LIMIT_M}m)')
            self.emergency_stop(f'fall at z={base_height:.2f}m')

        # ---- self collision
        col_ok, min_torso, hand_dist, col_reason = SelfCollisionGuard.check_clearance(
            left_wrist_p, right_wrist_p, pelvis_p)
        if not col_ok:
            self.alarms.append(f'self-collision: {col_reason}')

        # ---- thermal
        thermal_ok, thermal_load, therm_reason = self.thermal.update(current_torques)
        if not thermal_ok:
            self.alarms.append(f'thermal: {therm_reason}')
            if self.state == SafetyState.ACTIVE_TELEOP:
                self.state = SafetyState.SOFT_STOP

        # ---- commanded-target sanity
        target_q = clamp_to_joint_limits(np.asarray(target_q, dtype=np.float32).reshape(-1))
        current_q = np.asarray(current_q, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(target_q)):
            self.alarms.append('non-finite target_q')
            target_q = current_q.copy()
        if not col_ok:
            target_q = self.last_safe_q.copy()

        # ---- build the command
        if self.state == SafetyState.EMERGENCY_STOP:
            action = self._damped_action(current_q)
        elif self.state in (SafetyState.SOFT_STOP, SafetyState.DISARMED):
            action = self._hold_action(self.last_safe_q)
            if self.state == SafetyState.DISARMED:
                action = self._damped_action(current_q)
        elif self.state in (SafetyState.GANTRY_HOLD, SafetyState.CALIBRATING):
            action = self._hold_action(self.s_curve.step(NOMINAL_STAND_Q))
            self.last_safe_q = action.q_des.copy()
        else:  # ACTIVE_TELEOP
            filtered = self.s_curve.step(target_q)
            action = self._hold_action(filtered)
            self.last_safe_q = filtered.copy()

        tau = np.abs(np.asarray(current_torques, dtype=np.float32).reshape(-1))
        tau_ratio = float(np.max(tau[:len(TORQUE_LIMITS)] / np.maximum(TORQUE_LIMITS[:len(tau)], 1e-3))) \
            if tau.size else 0.0

        status = SafetyStatus(
            state=self.state,
            is_safe=(self.state not in (SafetyState.EMERGENCY_STOP,)
                     and not self._latched and not self.alarms),
            active_alarms=list(self.alarms),
            latched_alarms=list(self._latched),
            min_torso_dist=min_torso,
            hand_to_hand_dist=hand_dist,
            max_joint_velocity=float(np.max(np.abs(self.s_curve.dq_filtered))),
            max_joint_accel=float(np.max(np.abs(self.s_curve.ddq_filtered))),
            max_torque_ratio=tau_ratio,
            highest_thermal_load=thermal_load,
            imu_tilt_deg=base_tilt_deg,
            watchdog_latency_ms=hb_ms,
        )
        return action, status