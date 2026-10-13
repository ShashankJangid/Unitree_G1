from __future__ import annotations
import abc
import time
import threading
from typing import Dict, Optional, Tuple, Callable
import numpy as np
from g1_constants import (
    DEFAULT_KD, DEFAULT_KP, NOMINAL_STAND_Q, ZERO_POSE_Q, NUM_BODY_MOTORS, NUM_MOTORS,
    ARM_SDK_ENABLE, LEG_JOINT_INDICES, LEFT_ARM_INDICES, RIGHT_ARM_INDICES, WAIST_INDICES,
    TOPIC_LOW_CMD, TOPIC_LOW_STATE, DDS_DOMAIN_SIM, DDS_DOMAIN_REAL,
    MOTOR_MODE_ENABLE, MOTOR_MODE_DISABLE, TORQUE_LIMITS
)

class G1Bridge(abc.ABC):

    @abc.abstractmethod
    def send_joint_targets(
        self,
        q_des: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        dq_des: Optional[np.ndarray] = None,
        tau_ff: Optional[np.ndarray] = None
    ) -> None:
        pass

    @abc.abstractmethod
    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def get_imu_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def is_hardware_active(self) -> bool:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

    @abc.abstractmethod
    def enable_arm_sdk(self, enable: bool = True) -> bool:
        pass

    @abc.abstractmethod
    def release_motion_mode(self) -> bool:
        pass


class G1SimBridge(G1Bridge):

    def __init__(self, robot=None, num_motors: int = NUM_BODY_MOTORS):
        from g1_robot import G1Robot
        self.num_motors = num_motors
        if robot is None:
            self.robot = G1Robot(fix_base=True, num_motors=num_motors)
        else:
            self.robot = robot

    def send_joint_targets(
        self,
        q_des: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        dq_des: Optional[np.ndarray] = None,
        tau_ff: Optional[np.ndarray] = None
    ) -> None:
        if kp is not None:
            self.robot.kp[:len(kp)] = kp
        if kd is not None:
            self.robot.kd[:len(kd)] = kd
        self.robot.step_physics(q_des, dq_des, tau_ff)

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.robot.get_joint_positions(), self.robot.get_joint_velocities())

    def get_imu_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.robot.get_imu_data()

    def is_hardware_active(self) -> bool:
        return False

    def close(self) -> None:
        pass

    def enable_arm_sdk(self, enable: bool = True) -> bool:
        return True

    def release_motion_mode(self) -> bool:
        return True


class G1RealHardwareBridge(G1Bridge):
    MAX_JOINT_VELOCITY = 1.2
    CONTROL_DT = 0.002
    POLICY_DT = 0.02

    def __init__(
        self,
        iface: str = 'Ethernet',
        domain_id: int = DDS_DOMAIN_REAL,
        num_motors: int = NUM_BODY_MOTORS,
        auto_enable_arm_sdk: bool = True
    ):
        self.iface = iface
        self.domain_id = domain_id
        self.num_motors = num_motors
        self.auto_enable_arm_sdk = auto_enable_arm_sdk

        self.is_connected = False
        self.is_streaming = False
        self._running = False

        self._latest_q = NOMINAL_STAND_Q[:num_motors].copy()
        self._latest_dq = np.zeros(num_motors, dtype=np.float32)
        self._latest_tau = np.zeros(num_motors, dtype=np.float32)
        self._last_cmd_q = NOMINAL_STAND_Q[:num_motors].copy()
        self._last_send_time = time.time()
        self._has_received_state = False
        self._state_lock = threading.Lock()

        self._imu_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._imu_angvel = np.zeros(3, dtype=np.float32)
        self._imu_linacc = np.zeros(3, dtype=np.float32)

        self._policy_thread: Optional[threading.Thread] = None
        self._control_thread: Optional[threading.Thread] = None
        self._policy_callback: Optional[Callable] = None
        self._target_q = NOMINAL_STAND_Q[:num_motors].copy()
        self._target_kp = DEFAULT_KP[:num_motors].copy()
        self._target_kd = DEFAULT_KD[:num_motors].copy()
        self._target_dq = np.zeros(num_motors, dtype=np.float32)
        self._target_tau_ff = np.zeros(num_motors, dtype=np.float32)
        self._cmd_lock = threading.Lock()

        self._crc = None
        self._low_cmd = None
        self._pub = None
        self._sub = None
        self._motion_switcher = None

        print(f"[G1Hardware] Initializing CycloneDDS on '{iface}' (Domain: {domain_id})...")
        self._init_dds()

    def _init_dds(self):
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.utils.crc import CRC
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

            ChannelFactoryInitialize(self.domain_id, self.iface)

            self._pub = ChannelPublisher(TOPIC_LOW_CMD, LowCmd_)
            self._pub.Init()

            self._sub = ChannelSubscriber(TOPIC_LOW_STATE, LowState_)
            self._sub.Init(self._state_callback, 10)

            self._low_cmd = unitree_hg_msg_dds__LowCmd_()
            for i, mc in enumerate(self._low_cmd.motor_cmd):
                if i >= self.num_motors:
                    break
                mc.mode = MOTOR_MODE_DISABLE
                mc.kp = float(DEFAULT_KP[i])
                mc.kd = float(DEFAULT_KD[i])
                mc.q = float(NOMINAL_STAND_Q[i])
                mc.dq = 0.0
                mc.tau = 0.0

            if len(self._low_cmd.motor_cmd) > ARM_SDK_ENABLE:
                self._low_cmd.motor_cmd[ARM_SDK_ENABLE].q = 0.0

            self._crc = CRC()
            self.is_connected = True

            self._motion_switcher = MotionSwitcherClient()
            self._motion_switcher.SetTimeout(10.0)
            self._motion_switcher.Init()

            print('[G1Hardware] DDS channels ready.')
        except Exception as e:
            print(f'[G1Hardware] Warning: Could not initialize Unitree DDS ({e}).')
            self.is_connected = False

    def _state_callback(self, msg):
        try:
            with self._state_lock:
                for i in range(min(self.num_motors, len(msg.motor_state))):
                    self._latest_q[i] = msg.motor_state[i].q
                    self._latest_dq[i] = msg.motor_state[i].dq
                    self._latest_tau[i] = msg.motor_state[i].tau_est
                if not self._has_received_state:
                    self._has_received_state = True
                    self._last_cmd_q[:] = self._latest_q[:]
        except Exception as e:
            print(f"[G1Hardware] State callback error: {e}")

    def _imu_callback(self, msg):
        try:
            self._imu_quat = np.array([msg.quaternion[0], msg.quaternion[1], msg.quaternion[2], msg.quaternion[3]], dtype=np.float32)
            self._imu_angvel = np.array([msg.gyroscope[0], msg.gyroscope[1], msg.gyroscope[2]], dtype=np.float32)
            self._imu_linacc = np.array([msg.accelerometer[0], msg.accelerometer[1], msg.accelerometer[2]], dtype=np.float32)
        except Exception:
            pass

    def wait_for_robot(self, timeout: float = 5.0) -> bool:
        if not self.is_connected:
            return False
        start = time.time()
        print('[G1Hardware] Waiting for handshake from G1 robot...')
        while time.time() - start < timeout:
            with self._state_lock:
                if self._has_received_state:
                    print('[G1Hardware] Handshake confirmed! Robot telemetry active.')
                    return True
            time.sleep(0.05)
        print('[G1Hardware] No robot feedback received. Operating in offline/safe mode.')
        return False

    def release_motion_mode(self) -> bool:
        if not self.is_connected or self._motion_switcher is None:
            return False
        try:
            print('[G1Hardware] Releasing motion mode (sport service)...')
            ret = self._motion_switcher.ReleaseMode()
            if ret == 0:
                print('[G1Hardware] Motion mode released successfully.')
                return True
            else:
                print(f'[G1Hardware] Failed to release motion mode: {ret}')
                return False
        except Exception as e:
            print(f'[G1Hardware] Error releasing motion mode: {e}')
            return False

    def ramp_arm_sdk_weight(self, target_weight: float = 1.0, duration: float = 1.0, steps: int = 50) -> None:
        """Smoothly ramp Arm SDK blending weight on motor slot 29 (kNotUsedJoint0) matching official Unitree SDK2 pattern."""
        if not self.is_connected:
            return
        try:
            from g1_constants import ARM_SDK_WEIGHT_INDEX
            if len(self._low_cmd.motor_cmd) > ARM_SDK_WEIGHT_INDEX:
                current_w = float(self._low_cmd.motor_cmd[ARM_SDK_WEIGHT_INDEX].q)
                dt = duration / max(1, steps)
                for w in np.linspace(current_w, target_weight, steps):
                    self._low_cmd.motor_cmd[ARM_SDK_WEIGHT_INDEX].q = float(w)
                    if self._crc is not None:
                        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
                    self._pub.Write(self._low_cmd)
                    time.sleep(dt)
                print(f"[G1Hardware] Arm SDK weight smoothly transitioned to {target_weight:.2f}.")
        except Exception as e:
            print(f"[G1Hardware] Error ramping arm_sdk weight: {e}")

    def enable_arm_sdk(self, enable: bool = True, smooth: bool = True) -> bool:
        if not self.is_connected:
            return False
        try:
            target_w = 1.0 if enable else 0.0
            if smooth:
                self.ramp_arm_sdk_weight(target_w, duration=0.8)
            else:
                from g1_constants import ARM_SDK_WEIGHT_INDEX
                if len(self._low_cmd.motor_cmd) > ARM_SDK_WEIGHT_INDEX:
                    self._low_cmd.motor_cmd[ARM_SDK_WEIGHT_INDEX].q = target_w
                    if self._crc is not None:
                        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
                    self._pub.Write(self._low_cmd)
            print(f'[G1Hardware] Arm SDK {"enabled" if enable else "disabled"}.')
            return True
        except Exception as e:
            print(f'[G1Hardware] Error setting arm_sdk: {e}')
        return False

    def _control_loop(self):
        """500 Hz control loop - publishes commands to hardware."""
        while self._running:
            loop_start = time.time()

            if self.is_streaming and self.is_connected:
                self._publish_commands()

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, self.CONTROL_DT - elapsed)
            time.sleep(sleep_time)

    def _policy_loop(self):
        """Policy rate loop - calls policy callback if set."""
        while self._running:
            loop_start = time.time()

            if self._policy_callback and self.is_streaming:
                try:
                    with self._state_lock:
                        q = self._latest_q.copy()
                        dq = self._latest_dq.copy()
                    self._policy_callback(q, dq)
                except Exception as e:
                    print(f"[G1Hardware] Policy callback error: {e}")

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, self.POLICY_DT - elapsed)
            time.sleep(sleep_time)

    def _publish_commands(self):
        now = time.time()
        dt = max(0.0001, now - self._last_send_time)
        self._last_send_time = now

        max_delta = self.MAX_JOINT_VELOCITY * dt
        with self._cmd_lock:
            q_clamped = np.clip(
                self._target_q,
                self._last_cmd_q - max_delta,
                self._last_cmd_q + max_delta
            )
            self._last_cmd_q[:] = q_clamped[:]
            target_kp = self._target_kp.copy()
            target_kd = self._target_kd.copy()
            target_dq = self._target_dq.copy()
            target_tau = self._target_tau_ff.copy()

        for i in range(min(self.num_motors, len(self._low_cmd.motor_cmd))):
            mc = self._low_cmd.motor_cmd[i]
            mc.q = float(q_clamped[i])
            mc.dq = float(target_dq[i])
            mc.kp = float(target_kp[i])
            mc.kd = float(target_kd[i])
            mc.tau = float(target_tau[i])
            mc.mode = MOTOR_MODE_ENABLE

        if len(self._low_cmd.motor_cmd) > ARM_SDK_ENABLE:
            self._low_cmd.motor_cmd[ARM_SDK_ENABLE].q = 1.0

        if self._crc is not None:
            self._low_cmd.crc = self._crc.Crc(self._low_cmd)

        try:
            self._pub.Write(self._low_cmd)
        except Exception as e:
            print(f"[G1Hardware] Publish error: {e}")

    def start_streaming(self, policy_callback: Optional[Callable] = None) -> bool:
        if not self.is_connected:
            print('[G1Hardware] Cannot start streaming: DDS not connected.')
            return False

        if not self._has_received_state:
            print('[G1Hardware] No state received yet. Call wait_for_robot() first.')
            return False

        print('[G1Hardware] Releasing motion mode before starting...')
        self.release_motion_mode()
        time.sleep(0.5)

        print('[G1Hardware] Enabling arm_sdk...')
        self.enable_arm_sdk(True)
        time.sleep(0.2)

        with self._cmd_lock:
            self._target_q[:] = self._latest_q[:]
            self._last_cmd_q[:] = self._latest_q[:]

        self._policy_callback = policy_callback
        self._running = True
        self.is_streaming = True

        self._control_thread = threading.Thread(target=self._control_loop, name="G1ControlLoop", daemon=True)
        self._control_thread.start()

        self._policy_thread = threading.Thread(target=self._policy_loop, name="G1PolicyLoop", daemon=True)
        self._policy_thread.start()

        print('\n' + '=' * 60)
        print('>>> HARDWARE STREAMING ACTIVE: MOTORS ENGAGED! <<<')
        print('=' * 60)
        return True

    def stop_streaming(self):
        self.is_streaming = False
        self._running = False

        if self._control_thread:
            self._control_thread.join(timeout=1.0)
        if self._policy_thread:
            self._policy_thread.join(timeout=1.0)

        self.enable_arm_sdk(False)

        print('\n' + '=' * 60)
        print('>>> HARDWARE STREAMING DISENGAGED (STANDBY) <<<')
        print('=' * 60)

    def toggle_streaming(self, policy_callback: Optional[Callable] = None) -> bool:
        if self.is_streaming:
            self.stop_streaming()
            return False
        else:
            return self.start_streaming(policy_callback)

    def is_hardware_active(self) -> bool:
        return self.is_connected and self.is_streaming

    def send_joint_targets(
        self,
        q_des: np.ndarray,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        dq_des: Optional[np.ndarray] = None,
        tau_ff: Optional[np.ndarray] = None
    ) -> None:
        if not self.is_streaming:
            return

        with self._cmd_lock:
            self._target_q[:] = q_des[:self.num_motors]
            if kp is not None:
                self._target_kp[:len(kp)] = kp
            if kd is not None:
                self._target_kd[:len(kd)] = kd
            if dq_des is not None:
                self._target_dq[:len(dq_des)] = dq_des
            if tau_ff is not None:
                self._target_tau_ff[:len(tau_ff)] = tau_ff

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        with self._state_lock:
            return (self._latest_q.copy(), self._latest_dq.copy())

    def get_imu_state(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self._imu_quat.copy(), self._imu_angvel.copy(), self._imu_linacc.copy())

    def close(self) -> None:
        self.stop_streaming()
        self.is_connected = False

    def emergency_stop(self):
        """Emergency stop - immediately disable motors and stop streaming."""
        print('\n[G1Hardware] !!! EMERGENCY STOP TRIGGERED !!!')
        self.stop_streaming()
        if self.is_connected and self._low_cmd is not None:
            for i in range(min(self.num_motors, len(self._low_cmd.motor_cmd))):
                self._low_cmd.motor_cmd[i].mode = MOTOR_MODE_DISABLE
                self._low_cmd.motor_cmd[i].kp = 0.0
                self._low_cmd.motor_cmd[i].kd = 0.0
            if len(self._low_cmd.motor_cmd) > ARM_SDK_ENABLE:
                self._low_cmd.motor_cmd[ARM_SDK_ENABLE].q = 0.0
            if self._crc:
                self._low_cmd.crc = self._crc.Crc(self._low_cmd)
            try:
                self._pub.Write(self._low_cmd)
            except Exception:
                pass


def create_g1_bridge(
    mode: str = 'sim',
    iface: str = 'Ethernet',
    domain_id: Optional[int] = None,
    robot=None,
    num_motors: int = NUM_BODY_MOTORS
) -> G1Bridge:
    if domain_id is None:
        domain_id = DDS_DOMAIN_REAL if mode.lower() == 'hardware' else DDS_DOMAIN_SIM

    if mode.lower() == 'hardware':
        return G1RealHardwareBridge(iface=iface, domain_id=domain_id, num_motors=num_motors)
    return G1SimBridge(robot=robot, num_motors=num_motors)
