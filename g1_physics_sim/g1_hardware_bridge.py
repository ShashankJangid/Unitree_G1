from __future__ import annotations
import abc
import time
from typing import Dict, Optional, Tuple
import numpy as np
from g1_constants import DEFAULT_KD, DEFAULT_KP, NOMINAL_STAND_Q, NUM_MOTORS, L_SHOULDER_PITCH, R_WRIST_YAW

class G1Bridge(abc.ABC):

    @abc.abstractmethod
    def send_joint_targets(self, q_des: np.ndarray, kp: Optional[np.ndarray]=None, kd: Optional[np.ndarray]=None) -> None:
        pass

    @abc.abstractmethod
    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abc.abstractmethod
    def is_hardware_active(self) -> bool:
        pass

    @abc.abstractmethod
    def close(self) -> None:
        pass

class G1SimBridge(G1Bridge):

    def __init__(self, robot=None):
        if robot is None:
            from g1_robot import G1Robot
            self.robot = G1Robot(fix_base=True)
        else:
            self.robot = robot

    def send_joint_targets(self, q_des: np.ndarray, kp: Optional[np.ndarray]=None, kd: Optional[np.ndarray]=None) -> None:
        if kp is not None:
            self.robot.kp = kp
        if kd is not None:
            self.robot.kd = kd
        self.robot.step_physics(q_des)

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.robot.get_joint_positions(), self.robot.get_joint_velocities())

    def is_hardware_active(self) -> bool:
        return False

    def close(self) -> None:
        pass

class G1RealHardwareBridge(G1Bridge):
    MAX_JOINT_VELOCITY = 1.2

    def __init__(self, iface: str='Ethernet', domain_id: int=0):
        self.iface = iface
        self.domain_id = domain_id
        self.is_connected = False
        self.is_streaming = False
        self._latest_q = NOMINAL_STAND_Q.copy()
        self._latest_dq = np.zeros(NUM_MOTORS, dtype=np.float32)
        self._last_cmd_q = NOMINAL_STAND_Q.copy()
        self._last_send_time = time.time()
        self._has_received_state = False
        print(f"[G1Hardware] Initializing CycloneDDS on '{iface}' (Domain: {domain_id})...")
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            ChannelFactoryInitialize(self.domain_id, self.iface)
            self._pub = ChannelPublisher('rt/arm_sdk', LowCmd_)
            self._pub.Init()
            self._sub = ChannelSubscriber('rt/lowstate', LowState_)
            self._sub.Init(self._state_callback, 10)
            self._cmd = unitree_hg_msg_dds__LowCmd_()
            for i, mc in enumerate(self._cmd.motor_cmd):
                mc.mode = 0
                mc.kp = float(DEFAULT_KP[i]) if i < NUM_MOTORS else 40.0
                mc.kd = float(DEFAULT_KD[i]) if i < NUM_MOTORS else 1.0
                mc.q = float(NOMINAL_STAND_Q[i]) if i < NUM_MOTORS else 0.0
            if len(self._cmd.motor_cmd) > 29:
                self._cmd.motor_cmd[29].q = 1.0
            try:
                from unitree_sdk2py.utils.crc import CRC
                self._crc = CRC()
            except Exception:
                self._crc = None
            self.is_connected = True
            print('[G1Hardware] DDS channel ready.')
        except Exception as e:
            print(f'[G1Hardware] Warning: Could not initialize Unitree DDS ({e}).')
            self.is_connected = False

    def _state_callback(self, msg):
        try:
            for i in range(min(NUM_MOTORS, len(msg.motor_state))):
                self._latest_q[i] = msg.motor_state[i].q
                self._latest_dq[i] = msg.motor_state[i].dq
            if not self._has_received_state:
                self._has_received_state = True
                self._last_cmd_q[:] = self._latest_q[:]
        except Exception:
            pass

    def wait_for_robot(self, timeout: float=3.0) -> bool:
        if not self.is_connected:
            return False
        start = time.time()
        print('[G1Hardware] Waiting for handshake from G1 robot...')
        while time.time() - start < timeout:
            if self._has_received_state:
                print('[G1Hardware] Handshake confirmed! Robot telemetry active.')
                return True
            time.sleep(0.05)
        print('[G1Hardware] No robot feedback received. Operating in offline/safe mode.')
        return False

    def toggle_streaming(self) -> bool:
        if not self.is_connected:
            print('[G1Hardware] Cannot enable streaming: DDS not connected.')
            self.is_streaming = False
            return False
        self.is_streaming = not self.is_streaming
        if self.is_streaming:
            self._last_cmd_q[:] = self._latest_q[:]
            self._last_send_time = time.time()
            print('\n' + '=' * 60)
            print('>>> HARDWARE STREAMING ACTIVE: MOTORS ENGAGED! <<<')
            print('=' * 60)
        else:
            print('\n' + '=' * 60)
            print('>>> HARDWARE STREAMING DISENGAGED (STANDBY) <<<')
            print('=' * 60)
        return self.is_streaming

    def is_hardware_active(self) -> bool:
        return self.is_connected and self.is_streaming

    def send_joint_targets(self, q_des: np.ndarray, kp: Optional[np.ndarray]=None, kd: Optional[np.ndarray]=None) -> None:
        if not self.is_connected or not self.is_streaming:
            return
        now = time.time()
        dt = max(0.0001, now - self._last_send_time)
        self._last_send_time = now
        max_delta = self.MAX_JOINT_VELOCITY * dt
        q_clamped = np.clip(q_des, self._last_cmd_q - max_delta, self._last_cmd_q + max_delta)
        self._last_cmd_q[:] = q_clamped[:]
        for i in range(min(NUM_MOTORS, len(self._cmd.motor_cmd))):
            self._cmd.motor_cmd[i].q = float(q_clamped[i])
            if kp is not None:
                self._cmd.motor_cmd[i].kp = float(kp[i])
            if kd is not None:
                self._cmd.motor_cmd[i].kd = float(kd[i])
        if len(self._cmd.motor_cmd) > 29:
            self._cmd.motor_cmd[29].q = 1.0
        if self._crc is not None:
            if hasattr(self._crc, 'Crc'):
                self._cmd.crc = self._crc.Crc(self._cmd)
            elif hasattr(self._crc, 'calculate_crc'):
                self._cmd.crc = self._crc.calculate_crc(self._cmd)
        self._pub.Write(self._cmd)

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._latest_q.copy(), self._latest_dq.copy())

    def close(self) -> None:
        self.is_streaming = False
        self.is_connected = False

def create_g1_bridge(mode: str='sim', iface: str='Ethernet', robot=None) -> G1Bridge:
    if mode.lower() == 'hardware':
        return G1RealHardwareBridge(iface=iface)
    return G1SimBridge(robot=robot)
