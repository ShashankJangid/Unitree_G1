from __future__ import annotations
import abc
import time
import threading
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np

try:  # package and flat-script imports both work
    from .g1_constants import (    OFFICIAL_KP, OFFICIAL_KD, NOMINAL_STAND_Q, ZERO_POSE_Q,
    NUM_BODY_MOTORS,
    ARM_SDK_ENABLE, TOPIC_LOW_CMD, TOPIC_LOW_STATE, TOPIC_ARM_SDK,
    TOPIC_IMU_TORSO, TOPIC_HAND_SDK,
    DDS_DOMAIN_SIM, DDS_DOMAIN_REAL,
    MODE_PR, MODE_AB, MOTOR_MODE_ENABLE, MOTOR_MODE_DISABLE,
    LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, WAIST_JOINTS, LEG_JOINTS,
    ARM_SDK_JOINTS_13, ARM_SDK_JOINTS_17,
    MAX_JOINT_VELOCITY_FULL, ARM_SDK_MAX_JOINT_VELOCITY,
    ARM_SDK_WEIGHT_RATE, ARM_SDK_CONTROL_DT, LOWCMD_CONTROL_DT,
    TORQUE_LIMITS, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX, JOINT_LIMIT_MARGIN)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (OFFICIAL_KP, OFFICIAL_KD, NOMINAL_STAND_Q, ZERO_POSE_Q,
NUM_BODY_MOTORS,
ARM_SDK_ENABLE, TOPIC_LOW_CMD, TOPIC_LOW_STATE, TOPIC_ARM_SDK,
TOPIC_IMU_TORSO, TOPIC_HAND_SDK,
DDS_DOMAIN_SIM, DDS_DOMAIN_REAL,
MODE_PR, MODE_AB, MOTOR_MODE_ENABLE, MOTOR_MODE_DISABLE,
LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS, WAIST_JOINTS, LEG_JOINTS,
ARM_SDK_JOINTS_13, ARM_SDK_JOINTS_17,
MAX_JOINT_VELOCITY_FULL, ARM_SDK_MAX_JOINT_VELOCITY,
ARM_SDK_WEIGHT_RATE, ARM_SDK_CONTROL_DT, LOWCMD_CONTROL_DT,
TORQUE_LIMITS, JOINT_LIMITS_MIN, JOINT_LIMITS_MAX, JOINT_LIMIT_MARGIN)

try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_, IMUState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.utils.thread import RecurrentThread
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
    SDK_AVAILABLE = True
    SDK_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    SDK_AVAILABLE = False
    SDK_IMPORT_ERROR = _e
    LowCmd_ = LowState_ = IMUState_ = unitree_hg_msg_dds__LowCmd_ = None
    CRC = RecurrentThread = MotionSwitcherClient = None
    ChannelFactoryInitialize = ChannelPublisher = ChannelSubscriber = None


class G1Bridge(abc.ABC):
    """Abstract sim/real bridge."""

    @abc.abstractmethod
    def send_joint_targets(self, q_des, kp=None, kd=None, dq_des=None, tau_ff=None) -> None: ...
    @abc.abstractmethod
    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]: ...
    @abc.abstractmethod
    def get_imu_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
    @abc.abstractmethod
    def is_hardware_active(self) -> bool: ...
    @abc.abstractmethod
    def close(self) -> None: ...
    @abc.abstractmethod
    def emergency_stop(self) -> None: ...
    @abc.abstractmethod
    def tick(self) -> None: ...
    # ABC helper kept for backwards compatibility with older callers
    def enable_arm_sdk(self, enable: bool = True) -> bool:
        return False
    def release_motion_mode(self) -> bool:
        return False


class G1SimBridge(G1Bridge):
    """MuJoCo-backed bridge. Owns no threads; the caller drives tick()."""

    def __init__(self, robot=None, num_motors: int = NUM_BODY_MOTORS):
        from g1_robot import G1Robot
        self.num_motors = NUM_BODY_MOTORS
        self.robot = robot if robot is not None else G1Robot(fix_base=True)
        self._target_q = NOMINAL_STAND_Q.copy()
        self._target_dq = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
        self._target_tau_ff = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)

    def send_joint_targets(self, q_des, kp=None, kd=None, dq_des=None, tau_ff=None) -> None:
        n = min(len(q_des), NUM_BODY_MOTORS)
        self._target_q[:n] = q_des[:n]
        if kp is not None:
            self.robot.kp[:min(len(kp), NUM_BODY_MOTORS)] = kp[:NUM_BODY_MOTORS]
        if kd is not None:
            self.robot.kd[:min(len(kd), NUM_BODY_MOTORS)] = kd[:NUM_BODY_MOTORS]
        if dq_des is not None:
            self._target_dq[:min(len(dq_des), NUM_BODY_MOTORS)] = dq_des[:NUM_BODY_MOTORS]
        if tau_ff is not None:
            self._target_tau_ff[:min(len(tau_ff), NUM_BODY_MOTORS)] = tau_ff[:NUM_BODY_MOTORS]

    def tick(self) -> None:
        self.robot.step_physics(self._target_q, self._target_dq, self._target_tau_ff)

    def get_joint_states(self):
        return (self.robot.get_joint_positions(), self.robot.get_joint_velocities())

    def get_imu_state(self):
        return self.robot.get_imu_data()

    def is_hardware_active(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def emergency_stop(self) -> None:
        self._target_q[:] = self.robot.get_joint_positions()
        self._target_dq[:] = 0.0
        self._target_tau_ff[:] = 0.0


class G1RealHardwareBridge(G1Bridge):
    """
    Real-robot bridge speaking the genuine Unitree G1 DDS protocol.

    Two control paths, both SDK-faithful:
      * FULL BODY : publish LowCmd_ to 'rt/lowcmd'   @ 500 Hz
      * ARM ONLY  : publish LowCmd_ to 'rt/arm_sdk'  @  50 Hz
                    (arm_sdk only accepts 13/17 upper-body joints; legs are
                     deliberately never touched on this path)

    Protocol details that MUST match the robot:
      * LowCmd_.mode_pr    (0 = PR series, 1 = AB parallel) for ankle pairs
      * LowCmd_.mode_machine mirrored from LowState_.mode_machine (robot variant)
      * motor_cmd[i].mode  = 1 to enable, 0 to disable
      * CRC over the whole LowCmd_/LowState_ struct via unitree_sdk2py.utils.crc.CRC
      * MotionSwitcher ReleaseMode() before any low-level publishing
      * arm_sdk enable/disable is a WEIGHT RAMP on motor_cmd[29].q, never a hard 0/1
    """

    def __init__(
        self,
        iface: str = 'Ethernet',
        domain_id: int = DDS_DOMAIN_REAL,
        num_motors: int = NUM_BODY_MOTORS,
        control_path: str = 'full',
        auto_release_motion_mode: bool = True,
        enable_watchdog: bool = True,
    ):
        self.iface = iface
        self.domain_id = domain_id
        self.num_motors = num_motors if num_motors <= NUM_BODY_MOTORS else NUM_BODY_MOTORS
        if control_path not in ('full', 'arm'):
            raise ValueError("control_path must be 'full' (rt/lowcmd) or 'arm' (rt/arm_sdk)")
        self.control_path = control_path
        self.auto_release_motion_mode = auto_release_motion_mode
        self.enable_watchdog = enable_watchdog

        if control_path == 'arm':
            self.publish_topic = TOPIC_ARM_SDK
            self.control_dt = ARM_SDK_CONTROL_DT
            self.max_joint_velocity = ARM_SDK_MAX_JOINT_VELOCITY
            self.controlled_indices = list(ARM_SDK_JOINTS_17)
        else:
            self.publish_topic = TOPIC_LOW_CMD
            self.control_dt = LOWCMD_CONTROL_DT
            self.max_joint_velocity = MAX_JOINT_VELOCITY_FULL
            self.controlled_indices = list(range(NUM_BODY_MOTORS))

        self.is_connected = False
        self.is_streaming = False
        self._running = False

        self._latest_q = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
        self._latest_dq = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
        self._latest_tau = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
        self._latest_motor_state = np.zeros(NUM_BODY_MOTORS, dtype=np.int32)
        self._last_cmd_q = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
        self._last_send_time = time.time()
        self._has_received_state = False
        self._state_lock = threading.Lock()

# mode_machine identifies the robot variant and MUST be mirrored from
        # LowState_.mode_machine; 0 is only the pre-handshake placeholder. A
        # stale or wrong variant makes the robot decode the whole frame with the
        # wrong joint map, so it is never taken from user config.
        self.mode_machine = 0
        self.mode_pr = MODE_PR

        self._imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._imu_rpy = np.zeros(3, dtype=np.float32)
        self._imu_angvel = np.zeros(3, dtype=np.float32)
        self._imu_linacc = np.zeros(3, dtype=np.float32)
        self._imu_lock = threading.Lock()

        # wireless remote / E-stop
        self._remote_buttons = np.zeros(40, dtype=np.uint8)
        self._estop_pressed = False
        self._estop_lock = threading.Lock()

        self._target_q = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
        self._target_kp = OFFICIAL_KP.copy()
        self._target_kd = OFFICIAL_KD.copy()
        self._target_dq = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
        self._target_tau_ff = np.zeros(NUM_BODY_MOTORS, dtype=np.float32)
        self._target_mode = np.full(NUM_BODY_MOTORS, MOTOR_MODE_ENABLE, dtype=np.int32)
        self._cmd_lock = threading.Lock()

        # arm_sdk weight ramp state
        self._weight = 0.0
        self._weight_target = 0.0

        self._policy_callback: Optional[Callable] = None
        self._control_thread: Optional[threading.Thread] = None

        self._crc = None
        self._low_cmd = None
        self._pub = None
        self._sub = None
        self._imu_sub = None
        self._remote_sub = None
        self._motion_switcher = None
        self._motion_released = False

        self._state_age_at_start = 0.0
        self._publish_errors = 0
        self._crc_failures = 0
        self._last_state_rx = 0.0

        print(f"[G1Hardware] path={self.control_path} iface='{self.iface}' domain={self.domain_id} "
              f"topic={self.publish_topic} rate={1.0 / self.control_dt:.0f}Hz")
        self._init_dds()

    # ------------------------------------------------------------------ setup
    def _init_dds(self) -> None:
        if not SDK_AVAILABLE:
            print(f"[G1Hardware] unitree_sdk2py unavailable ({SDK_IMPORT_ERROR}). Offline mode.")
            return
        try:
            ChannelFactoryInitialize(self.domain_id, self.iface)

            self._pub = ChannelPublisher(self.publish_topic, LowCmd_)
            self._pub.Init()

            self._sub = ChannelSubscriber(TOPIC_LOW_STATE, LowState_)
            self._sub.Init(self._state_callback, 10)

            # Torso IMU (G1 only). IMUState_ exposes rpy/gyroscope/accelerometer,
            # NOT a quaternion - do not read msg.quaternion here.
            try:
                self._imu_sub = ChannelSubscriber(TOPIC_IMU_TORSO, IMUState_)
                self._imu_sub.Init(self._imu_callback, 10)
            except Exception as e:
                print(f"[G1Hardware] Torso IMU subscribe failed (non-fatal): {e}")

            self._low_cmd = unitree_hg_msg_dds__LowCmd_()
            self._crc = CRC()

            if self.auto_release_motion_mode:
                try:
                    self._motion_switcher = MotionSwitcherClient()
                    self._motion_switcher.SetTimeout(10.0)
                    self._motion_switcher.Init()
                except Exception as e:
                    print(f"[G1Hardware] MotionSwitcherClient unavailable: {e}")

            self.is_connected = True
            print('[G1Hardware] DDS ready.')
        except Exception as e:
            print(f'[G1Hardware] DDS init failed: {e}')
            self.is_connected = False

    # -------------------------------------------------------------- callbacks
    def _state_callback(self, msg) -> None:
        try:
            if self._crc is not None:
                if int(self._crc.Crc(msg)) != int(getattr(msg, 'crc', 0)):
                    self._crc_failures += 1
                    return  # drop corrupt frames
        except Exception:
            pass
        try:
            with self._state_lock:
                n = min(self.num_motors, len(msg.motor_state))
                for i in range(n):
                    ms = msg.motor_state[i]
                    self._latest_q[i] = ms.q
                    self._latest_dq[i] = ms.dq
                    self._latest_tau[i] = ms.tau_est
                    self._latest_motor_state[i] = ms.motorstate
                self.mode_machine = int(msg.mode_machine)
                self.mode_pr = int(msg.mode_pr)
                self._last_state_rx = time.time()
                try:
                    self._remote_buttons = np.frombuffer(
                        bytes(bytearray(msg.wireless_remote)), dtype=np.uint8
                    ).copy()
                except Exception:
                    pass
                if not self._has_received_state:
                    self._has_received_state = True
                    self._last_cmd_q[:] = self._latest_q[:]
        except Exception as e:
            print(f'[G1Hardware] state callback error: {e}')

    def state_age(self) -> float:
        """Seconds since the last valid LowState. Large value == link loss."""
        if self._last_state_rx == 0.0:
            return float('inf')
        return time.time() - self._last_state_rx

    def _imu_callback(self, msg) -> None:
        try:
            with self._imu_lock:
                self._imu_rpy = np.array(msg.rpy, dtype=np.float32)
                self._imu_angvel = np.array(msg.gyroscope, dtype=np.float32)
                self._imu_linacc = np.array(msg.accelerometer, dtype=np.float32)
                self._imu_quat = self._rpy_to_quat(self._imu_rpy)
        except Exception:
            pass

    @staticmethod
    def _rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
        """Roll-pitch-yaw (rad) -> quaternion (w, x, y, z)."""
        r, p, y = float(rpy[0]), float(rpy[1]), float(rpy[2])
        cr, sr = np.cos(r * 0.5), np.sin(r * 0.5)
        cp, sp = np.cos(p * 0.5), np.sin(p * 0.5)
        cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
        return np.array([
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ], dtype=np.float32)

    # ------------------------------------------------------------ motion mode
    def _check_motion_status(self) -> bool:
        """True while an onboard motion service still owns the motors."""
        if self._motion_switcher is None:
            return False
        try:
            form, name = '', ''
            self._motion_switcher.CheckMode(form, name)
            return bool(name)
        except Exception as e:
            print(f'[G1Hardware] CheckMode failed: {e}')
            return False

    def release_motion_mode(self, retries: int = 3, wait: float = 5.0) -> bool:
        if self._motion_switcher is None:
            print('[G1Hardware] MotionSwitcher unavailable; cannot release motion mode.')
            return False
        for attempt in range(retries):
            if not self._check_motion_status():
                self._motion_released = True
                return True
            print(f'[G1Hardware] Releasing motion mode (attempt {attempt + 1}/{retries})...')
            try:
                ret = self._motion_switcher.ReleaseMode()
            except Exception as e:
                print(f'[G1Hardware] ReleaseMode exception: {e}')
                ret = -1
            print(f'[G1Hardware] ReleaseMode -> {ret}')
            time.sleep(wait)
        ok = not self._check_motion_status()
        self._motion_released = ok
        return ok

    # -------------------------------------------------------------- streaming
    def wait_for_robot(self, timeout: float = 5.0) -> bool:
        if not self.is_connected:
            return False
        start = time.time()
        print('[G1Hardware] Waiting for LowState...')
        while time.time() - start < timeout:
            with self._state_lock:
                if self._has_received_state:
                    print('[G1Hardware] LowState received.')
                    return True
            time.sleep(0.02)
        print('[G1Hardware] No LowState received. Offline/safe mode.')
        return False

    def validate_state(self) -> List[str]:
        """Pre-flight checks against the real robot. Returns list of problems."""
        problems: List[str] = []
        if not self.is_connected:
            problems.append('DDS not connected')
            return problems
        if not self._has_received_state:
            problems.append('no LowState received - is the robot powered and on the right NIC?')
            return problems
        age = self.state_age()
        if age > 1.0:
            problems.append(f'stale LowState ({age:.2f}s old) - link loss')
        with self._state_lock:
            bad = [(i, int(s)) for i, s in enumerate(self._latest_motor_state) if s != 0]
            if bad:
                problems.append(f'motor fault codes present: {bad}')
            if not np.all(np.isfinite(self._latest_q)):
                problems.append('non-finite joint positions in LowState')
        if self._crc_failures:
            problems.append(f'{self._crc_failures} LowState CRC failures - check network/cable')
        if self.control_path == 'arm' and self.mode_machine == 0:
            print('[G1Hardware] NOTE: mode_machine == 0 (not yet reported). '
                  'Expected non-zero once LowState reports the robot variant.')
        if self._motion_switcher is not None and self._check_motion_status():
            problems.append('onboard motion service still active - ReleaseMode required')
        return problems

    def start_streaming(self, policy_callback: Optional[Callable] = None) -> bool:
        if not self.is_connected:
            print('[G1Hardware] Cannot stream: DDS not connected.')
            return False
        if not self._has_received_state:
            print('[G1Hardware] No LowState yet. Call wait_for_robot() first.')
            return False

        problems = self.validate_state()
        if problems:
            print('[G1Hardware] PRE-FLIGHT CHECK FAILED:')
            for p in problems:
                print(f'  - {p}')
            return False

        if self._motion_switcher is not None and not self._motion_released:
            if not self.release_motion_mode():
                print('[G1Hardware] Could not release motion mode; refusing to stream.')
                return False

        # Engage: for arm_sdk this is the weight ramp on motor_cmd[29].q
        self._weight_target = 1.0
        with self._cmd_lock:
            with self._state_lock:
                self._last_cmd_q[:] = self._latest_q[:]
                self._target_q[:] = self._latest_q[:]

        self.is_streaming = True
        self._last_send_time = time.time()
        print('\n' + '=' * 60)
        print(f'>>> STREAMING ARMED on {self.publish_topic} - MOTORS WILL ENGAGE <<<')
        print('=' * 60)
        return True

    def stop_streaming(self, ramp_down: bool = True) -> None:
        """Ramp the arm_sdk weight to zero (or latch motors off), then disarm."""
        if not self.is_connected or not self.is_streaming:
            self.is_streaming = False
            return
        if ramp_down and self.control_path == 'arm':
            t0 = time.time()
            span = max(1e-3, self._weight / ARM_SDK_WEIGHT_RATE)
            while time.time() - t0 < span:
                self._ramp_weight()
                try:
                    self._publish()
                except Exception:
                    break
                time.sleep(self.control_dt)
        else:
            self._weight = 0.0
            try:
                self._publish()
            except Exception:
                pass
        self._weight_target = 0.0
        self.is_streaming = False
        print('\n' + '=' * 60)
        print('>>> STREAMING DISENGAGED (STANDBY) <<<')
        print('=' * 60)

    def toggle_streaming(self, policy_callback: Optional[Callable] = None) -> bool:
        if self.is_streaming:
            self.stop_streaming()
            return False
        return self.start_streaming(policy_callback)

    def is_hardware_active(self) -> bool:
        return self.is_connected and self.is_streaming

    # ------------------------------------------------------------ control loop
    def _ramp_weight(self) -> None:
        step = ARM_SDK_WEIGHT_RATE * self.control_dt
        if self._weight < self._weight_target:
            self._weight = min(self._weight_target, self._weight + step)
        elif self._weight > self._weight_target:
            self._weight = max(self._weight_target, self._weight - step)

    def tick(self) -> None:
        """
        One control cycle. The OWNER (UnitreeG1Robot) calls this at control_dt.

        This bridge deliberately owns no threads: a single owner must drive the
        loop, otherwise two threads race on the same LowCmd_ and the weight ramp.
        """
        if not self.is_streaming:
            return
        with self._estop_lock:
            if self._estop_pressed:
                self.emergency_stop()
                return
        self._ramp_weight()
        try:
            self._publish()
        except Exception as e:
            self._publish_errors += 1
            if self._publish_errors % 50 == 1:
                print(f'[G1Hardware] publish error: {e}')

    def _resolve_targets(self) -> Tuple[np.ndarray, ...]:
        """
        Clamp to joint limits, then slew toward the target at no more than
        max_joint_velocity. The slew is measured against the PREVIOUSLY SENT
        command (self._last_cmd_q), exactly like the official examples - this
        is what bounds the commanded step per tick.
        """
        with self._cmd_lock:
            q_target = self._target_q.copy()
            kp = self._target_kp.copy()
            kd = self._target_kd.copy()
            dq = self._target_dq.copy()
            tau = self._target_tau_ff.copy()
            modes = self._target_mode.copy()

        idx = np.array(self.controlled_indices, dtype=np.int32)

        # 1) hard joint limits with margin
        q_target[idx] = np.clip(
            q_target[idx],
            JOINT_LIMITS_MIN[idx] + JOINT_LIMIT_MARGIN,
            JOINT_LIMITS_MAX[idx] - JOINT_LIMIT_MARGIN,
        )

        # 2) torque sanity before commanding gains
        tau[idx] = np.clip(tau[idx], -TORQUE_LIMITS[idx], TORQUE_LIMITS[idx])

        # 3) velocity slew vs previous command
        max_delta = self.max_joint_velocity * self.control_dt
        with self._state_lock:
            prev = self._last_cmd_q[idx].copy()
        stepped = prev + np.clip(q_target[idx] - prev, -max_delta, max_delta)

        with self._state_lock:
            self._last_cmd_q[idx] = stepped
        return stepped, kp, kd, dq, tau, modes

    def _publish(self) -> None:
        q_des, kp, kd, dq, tau, modes = self._resolve_targets()
        idx = self.controlled_indices

        cmd = self._low_cmd
        cmd.mode_pr = self.mode_pr
        cmd.mode_machine = self.mode_machine

        if self.control_path == 'arm':
            # arm_sdk path: only the 17 upper-body joints + weight ramp.
            # Legs are NEVER written here.
            for j in idx:
                i = int(j)
                mc = cmd.motor_cmd[i]
                mc.mode = MOTOR_MODE_ENABLE
                mc.q = float(q_des[i])
                mc.dq = float(dq[i])
                mc.kp = float(kp[i])
                mc.kd = float(kd[i])
                mc.tau = float(tau[i])
            cmd.motor_cmd[ARM_SDK_ENABLE].q = float(self._weight)
        else:
            # Full-body path: all 29 motors, explicit per-motor enable.
            for i in range(NUM_BODY_MOTORS):
                mc = cmd.motor_cmd[i]
                mc.mode = int(modes[i]) & 1
                mc.q = float(q_des[i])
                mc.dq = float(dq[i])
                mc.kp = float(kp[i])
                mc.kd = float(kd[i])
                mc.tau = float(tau[i])
                if self.enable_watchdog:
                    ms = int(self._latest_motor_state[i]) if self._latest_motor_state[i] else 0
                    if ms != 0 or not np.isfinite(q_des[i]):
                        # Faulted or NaN: back off to a damped hold.
                        mc.kp = 0.0
                        mc.kd = 1.0
                        mc.q = float(q_des[i]) if np.isfinite(q_des[i]) else 0.0
                        mc.tau = 0.0

        cmd.crc = self._crc.Crc(cmd)
        self._pub.Write(cmd)

    def send_joint_targets(self, q_des, kp=None, kd=None, dq_des=None, tau_ff=None) -> None:
        if not self.is_streaming:
            return
        idx = self.controlled_indices
        with self._cmd_lock:
            n = min(len(q_des), len(self._target_q))
            self._target_q[idx] = q_des[idx][:n]
            if kp is not None:
                self._target_kp[idx] = kp[:n]
            if kd is not None:
                self._target_kd[idx] = kd[:n]
            if dq_des is not None:
                self._target_dq[idx] = dq_des[:n]
            if tau_ff is not None:
                self._target_tau_ff[idx] = tau_ff[:n]

    def get_joint_states(self):
        with self._state_lock:
            return (self._latest_q.copy(), self._latest_dq.copy())

    def get_imu_state(self):
        with self._imu_lock:
            return (self._imu_quat.copy(), self._imu_angvel.copy(), self._imu_linacc.copy())

    def get_estop_pressed(self) -> bool:
        with self._estop_lock:
            return self._estop_pressed

    def close(self) -> None:
        self.stop_streaming()
        self.is_connected = False

    def emergency_stop(self) -> None:
        """Disable every motor immediately and stop the loop."""
        print('\n[G1Hardware] !!! EMERGENCY STOP !!!')
        self._running = False
        self.is_streaming = False
        self._weight = 0.0
        if not (self.is_connected and self._low_cmd is not None):
            return
        try:
            cmd = self._low_cmd
            if self.control_path == 'arm':
                cmd.motor_cmd[ARM_SDK_ENABLE].q = 0.0
            else:
                for i in range(NUM_BODY_MOTORS):
                    mc = cmd.motor_cmd[i]
                    mc.mode = MOTOR_MODE_DISABLE
                    mc.kp = 0.0
                    mc.kd = 0.0
                    mc.tau = 0.0
            cmd.crc = self._crc.Crc(cmd)
            self._pub.Write(cmd)
        except Exception as e:
            print(f'[G1Hardware] E-stop publish failed: {e}')


def create_g1_bridge(
    mode: str = 'sim',
    iface: str = 'Ethernet',
    domain_id: Optional[int] = None,
    robot=None,
    num_motors: int = NUM_BODY_MOTORS,
    control_path: str = 'full',
) -> G1Bridge:
    if domain_id is None:
        domain_id = DDS_DOMAIN_REAL if mode.lower() == 'hardware' else DDS_DOMAIN_SIM
    if mode.lower() == 'hardware':
        return G1RealHardwareBridge(
            iface=iface, domain_id=domain_id, num_motors=num_motors, control_path=control_path
        )
    return G1SimBridge(robot=robot, num_motors=num_motors)