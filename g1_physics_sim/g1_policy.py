"""
Single-owner runtime for the Unitree G1.

Thread ownership
----------------
Exactly one component drives the control loop. `UnitreeG1Robot` owns:
  * a control thread at control_dt (500 Hz on rt/lowcmd, 50 Hz on rt/arm_sdk)
  * a policy thread at policy_dt (default 50 Hz)

The bridge objects are deliberately thread-free and expose `tick()`. Two threads
publishing the same LowCmd_ would race on the CRC and the arm_sdk weight ramp,
so that arrangement is forbidden.

Safety
------
* Emergency stop is latched and requires an explicit reset.
* Joint velocity above limit demotes to SOFT_STOP (targets held), not E-stop;
  a transient overshoot must not kill a run.
* Motor fault codes (LowState_.motor_state[i].motorstate != 0) trigger E-stop.
"""
from __future__ import annotations

import abc
import copy
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

try:  # package and flat-script imports both work
    from .g1_constants import (    NUM_BODY_MOTORS, POLICY_FREQ, POLICY_DT, LOWCMD_CONTROL_DT,
    DDS_DOMAIN_SIM, DDS_DOMAIN_REAL, ARM_SDK_CONTROL_DT,
    OFFICIAL_KP, OFFICIAL_KD, NOMINAL_STAND_Q,
    LEG_JOINTS, WAIST_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS,
    MAX_JOINT_VELOCITY_FULL, FALL_HEIGHT_THRESHOLD,
    MOTOR_MODE_ENABLE, MOTOR_MODE_DISABLE)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (NUM_BODY_MOTORS, POLICY_FREQ, POLICY_DT, LOWCMD_CONTROL_DT,
DDS_DOMAIN_SIM, DDS_DOMAIN_REAL, ARM_SDK_CONTROL_DT,
OFFICIAL_KP, OFFICIAL_KD, NOMINAL_STAND_Q,
LEG_JOINTS, WAIST_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS,
MAX_JOINT_VELOCITY_FULL, FALL_HEIGHT_THRESHOLD,
MOTOR_MODE_ENABLE, MOTOR_MODE_DISABLE)
try:  # package and flat-script imports both work
    from .g1_hardware_bridge import (    G1Bridge,
    G1RealHardwareBridge,
    create_g1_bridge)
except ImportError:  # pragma: no cover - flat script layout
    from g1_hardware_bridge import (G1Bridge,
G1RealHardwareBridge,
create_g1_bridge)
try:  # package and flat-script imports both work
    from .g1_state import (    G1Action,
    G1Observation,
    G1State)
except ImportError:  # pragma: no cover - flat script layout
    from g1_state import (G1Action,
G1Observation,
G1State)


class Policy(abc.ABC):
    """Base class for anything that turns state into commands."""

    @abc.abstractmethod
    def reset(self, state: G1State) -> None: ...

    @abc.abstractmethod
    def step(self, state: G1State) -> G1Action: ...

    @abc.abstractmethod
    def get_observation(self, state: G1State) -> G1Observation: ...

    @property
    @abc.abstractmethod
    def action_dim(self) -> int: ...

    @property
    @abc.abstractmethod
    def obs_dim(self) -> int: ...

    def configure(self, config: Dict[str, Any]) -> None:
        pass

    def get_info(self) -> Dict[str, Any]:
        return {}


class UnitreeG1Robot:
    def __init__(
        self,
        mode: str = 'sim',
        iface: str = 'Ethernet',
        domain_id: Optional[int] = None,
        policy_rate: int = POLICY_FREQ,
        control_path: str = 'full',
        fix_base: bool = True,
        enable_elastic_band: bool = False,
        robot=None,
        max_joint_velocity: float = MAX_JOINT_VELOCITY_FULL,
        fall_height_threshold: float = FALL_HEIGHT_THRESHOLD,
    ):
        self.mode = mode.lower()
        self.control_path = control_path
        self.policy_rate = int(policy_rate)
        self.policy_dt = 1.0 / self.policy_rate
        self.control_dt = ARM_SDK_CONTROL_DT if control_path == 'arm' else LOWCMD_CONTROL_DT

        if self.mode not in ('sim', 'hardware'):
            raise ValueError("mode must be 'sim' or 'hardware'")
        if self.mode == 'sim' and domain_id is None:
            domain_id = DDS_DOMAIN_SIM
        elif self.mode == 'hardware' and domain_id is None:
            domain_id = DDS_DOMAIN_REAL

        self.bridge: G1Bridge = create_g1_bridge(
            mode=self.mode, iface=iface, domain_id=domain_id, robot=robot,
            num_motors=NUM_BODY_MOTORS, control_path=control_path,
        )

        self._running = False
        self._estop_latched = False
        self._policy: Optional[Policy] = None

        self._control_thread: Optional[threading.Thread] = None
        self._policy_thread: Optional[threading.Thread] = None

        self._state_lock = threading.Lock()
        self._latest_state: Optional[G1State] = None
        self._latest_action: Optional[G1Action] = None

        self._start_time = 0.0
        self._step_count = 0
        self._policy_errors = 0

        self.max_joint_velocity = float(max_joint_velocity)
        self.fall_height_threshold = float(fall_height_threshold)

        print(f'[UnitreeG1Robot] mode={self.mode} path={control_path} '
              f'iface={iface} domain={domain_id} '
              f'control={1.0 / self.control_dt:.0f}Hz policy={self.policy_rate}Hz')

    # ------------------------------------------------------------- lifecycle
    def set_policy(self, policy: Policy) -> None:
        self._policy = policy
        print(f'[UnitreeG1Robot] policy={type(policy).__name__}')

    def start(self, wait_for_robot: bool = True, timeout: float = 5.0) -> bool:
        if self._running:
            return True
        if self._estop_latched:
            print('[UnitreeG1Robot] E-stop is latched; call reset_estop() first.')
            return False

        self._start_time = time.time()
        self._step_count = 0

        if isinstance(self.bridge, G1RealHardwareBridge):
            if wait_for_robot and not self.bridge.wait_for_robot(timeout):
                print('[UnitreeG1Robot] no LowState; refusing to start.')
                return False
            problems = self.bridge.validate_state()
            if problems:
                print('[UnitreeG1Robot] PRE-FLIGHT FAILED:')
                for p in problems:
                    print(f'  - {p}')
                return False
            if not self.bridge.start_streaming():
                print('[UnitreeG1Robot] start_streaming() refused.')
                return False

        # Prime state so the policy sees a valid first observation.
        self._update_state()
        if self._policy is not None and self._latest_state is not None:
            try:
                self._policy.reset(self._latest_state)
            except Exception as e:
                print(f'[UnitreeG1Robot] policy.reset() failed: {e}')
                return False

        self._running = True
        self._control_thread = threading.Thread(target=self._control_loop, name='G1Ctrl', daemon=True)
        self._control_thread.start()
        if self._policy is not None:
            self._policy_thread = threading.Thread(target=self._policy_loop, name='G1Policy', daemon=True)
            self._policy_thread.start()
        print('[UnitreeG1Robot] running')
        return True

    def stop(self) -> None:
        """Graceful stop: ramp down, release, join. Does NOT latch E-stop."""
        if not self._running:
            self.bridge.close()
            return
        self._running = False
        if self._control_thread:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None
        if self._policy_thread:
            self._policy_thread.join(timeout=2.0)
            self._policy_thread = None
        self.bridge.close()
        print('[UnitreeG1Robot] stopped')

    def emergency_stop(self, reason: str = 'manual') -> None:
        """Latch E-stop, publish motor-disable immediately."""
        print(f'\n[UnitreeG1Robot] !!! EMERGENCY STOP ({reason}) !!!')
        self._estop_latched = True
        self._running = False
        try:
            self.bridge.emergency_stop()
        except Exception as e:
            print(f'[UnitreeG1Robot] e-stop publish error: {e}')

    def reset_estop(self) -> bool:
        """Clear a latched E-stop. Requires a fresh, valid pre-flight check."""
        if isinstance(self.bridge, G1RealHardwareBridge):
            problems = self.bridge.validate_state()
            if problems:
                print('[UnitreeG1Robot] cannot clear E-stop:')
                for p in problems:
                    print(f'  - {p}')
                return False
        self._estop_latched = False
        print('[UnitreeG1Robot] E-stop cleared')
        return True

    # ---------------------------------------------------------------- loops
    def _control_loop(self) -> None:
        period = self.control_dt
        next_t = time.perf_counter()
        while self._running and not self._estop_latched:
            self._update_state()
            try:
                self.bridge.tick()
            except Exception as e:
                print(f'[UnitreeG1Robot] bridge tick error: {e}')
                self.emergency_stop('bridge tick failure')
                break
            self._step_count += 1
            if (self._step_count & 0x3F) == 0:
                self._check_safety()
            next_t += period
            sleep_s = next_t - time.perf_counter()
            if sleep_s < 0:
                next_t = time.perf_counter()  # overran; resync
            else:
                time.sleep(sleep_s)

    def _policy_loop(self) -> None:
        period = self.policy_dt
        next_t = time.perf_counter()
        while self._running and not self._estop_latched:
            self._policy_step()
            next_t += period
            sleep_s = next_t - time.perf_counter()
            if sleep_s < 0:
                next_t = time.perf_counter()
            else:
                time.sleep(sleep_s)

    def _policy_step(self) -> None:
        if self._policy is None:
            return
        with self._state_lock:
            state = self._latest_state
        if state is None:
            return
        try:
            action = self._policy.step(copy.deepcopy(state))
        except Exception as e:
            self._policy_errors += 1
            print(f'[UnitreeG1Robot] policy.step error #{self._policy_errors}: {e}')
            if self._policy_errors >= 3:
                self.emergency_stop('repeated policy exceptions')
            return
        if not isinstance(action, G1Action):
            self.emergency_stop('policy returned a non-G1Action')
            return
        with self._state_lock:
            self._latest_action = action
        self.bridge.send_joint_targets(
            q_des=action.q_des, kp=action.kp, kd=action.kd,
            dq_des=action.dq_des, tau_ff=action.tau_ff,
        )

    # ---------------------------------------------------------------- state
    def _update_state(self) -> None:
        q, dq = self.bridge.get_joint_states()
        imu_quat, imu_angvel, imu_linacc = self.bridge.get_imu_state()

        q = np.asarray(q, dtype=np.float32).reshape(NUM_BODY_MOTORS)
        dq = np.asarray(dq, dtype=np.float32).reshape(NUM_BODY_MOTORS)

        state = G1State(
            timestamp=time.time() - self._start_time,
            q=q, dq=dq,
            imu_quat=imu_quat, imu_angvel=imu_angvel, imu_linacc=imu_linacc,
        )

        robot = getattr(self.bridge, 'robot', None)
        if robot is not None:
            state.base_pos, state.base_quat = robot.get_base_pose()
            state.base_linvel, state.base_angvel = robot.get_base_vel()
            state.left_wrist_pos = robot.get_hand_pos(False)
            state.right_wrist_pos = robot.get_hand_pos(True)
            state.left_wrist_quat = robot.get_hand_quat(False)
            state.right_wrist_quat = robot.get_hand_quat(True)
            state.left_foot_contact, state.right_foot_contact = robot.get_foot_contacts()
            state.tau_est = robot.get_joint_torques()
            state.mode_pr = robot.mode_pr

        hw = self.bridge if isinstance(self.bridge, G1RealHardwareBridge) else None
        if hw is not None:
            with hw._state_lock:
                state.tau_est = hw._latest_tau.copy()
                state.motor_state = hw._latest_motor_state.copy()
                state.mode_pr = hw.mode_pr
                state.mode_machine = hw.mode_machine
                state.crc_ok = hw._crc_failures == 0
            hw_angvel = hw._imu_angvel.copy()
            state.imu_angvel = hw_angvel

        with self._state_lock:
            self._latest_state = state

    # --------------------------------------------------------------- safety
    def _check_safety(self) -> None:
        with self._state_lock:
            state = self._latest_state
        if state is None:
            return

        if state.any_motor_faulted:
            self.emergency_stop(f'motor fault {state.get_faulted_motors()}')
            return

        if not np.all(np.isfinite(state.q)) or not np.all(np.isfinite(state.dq)):
            self.emergency_stop('non-finite joint feedback')
            return

        # Overspeed is a soft condition: hold position, do not latch E-stop.
        if np.any(np.abs(state.dq) > self.max_joint_velocity):
            with self._state_lock:
                if self._latest_action is not None:
                    self._latest_action.q_des[:] = state.q
                    self._latest_action.dq_des[:] = 0.0

        if self.mode == 'sim':
            robot = getattr(self.bridge, 'robot', None)
            if robot is not None and not self.bridge.is_hardware_active():
                base_h = robot.get_base_pose()[0][2]
                if base_h < self.fall_height_threshold:
                    self.emergency_stop(f'fall: base z={base_h:.2f}m')

    # -------------------------------------------------------------- getters
    def get_state(self) -> Optional[G1State]:
        with self._state_lock:
            return copy.deepcopy(self._latest_state) if self._latest_state else None

    def get_action(self) -> Optional[G1Action]:
        with self._state_lock:
            return copy.deepcopy(self._latest_action) if self._latest_action else None

    def is_running(self) -> bool:
        return self._running and not self._estop_latched

    def is_hardware_active(self) -> bool:
        return self.bridge.is_hardware_active()

    def get_info(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'control_path': self.control_path,
            'running': self._running,
            'hardware_active': self.is_hardware_active(),
            'step_count': self._step_count,
            'uptime': (time.time() - self._start_time) if self._start_time else 0.0,
            'estop_latched': self._estop_latched,
            'policy_errors': self._policy_errors,
            'max_joint_velocity': self.max_joint_velocity,
        }


def run_policy(
    policy: Policy,
    mode: str = 'sim',
    iface: str = 'Ethernet',
    domain_id: Optional[int] = None,
    policy_rate: int = POLICY_FREQ,
    control_path: str = 'full',
    duration: Optional[float] = None,
    fix_base: bool = True,
    enable_elastic_band: bool = False,
    robot=None,
) -> UnitreeG1Robot:
    runner = UnitreeG1Robot(
        mode=mode, iface=iface, domain_id=domain_id, policy_rate=policy_rate,
        control_path=control_path, fix_base=fix_base,
        enable_elastic_band=enable_elastic_band, robot=robot,
    )
    runner.set_policy(policy)
    if runner.start():
        try:
            if duration and duration > 0:
                time.sleep(duration)
            else:
                while runner.is_running():
                    time.sleep(0.5)
        except KeyboardInterrupt:
            print('\n[UnitreeG1Robot] interrupted')
        finally:
            runner.stop()
    return runner