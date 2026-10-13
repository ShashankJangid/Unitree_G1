from __future__ import annotations
import abc
import time
import threading
from typing import Dict, Optional, Any
import numpy as np

from g1_state import G1State, G1Action, G1Observation
from g1_hardware_bridge import G1Bridge, create_g1_bridge
from g1_robot import G1Robot
from g1_constants import (
    NUM_BODY_MOTORS, NUM_MOTORS, CONTROL_FREQ, PHYSICS_FREQ, DECIMATION,
    DDS_DOMAIN_SIM, DDS_DOMAIN_REAL, DEFAULT_KP, DEFAULT_KD
)


class Policy(abc.ABC):
    """Abstract base class for G1 policies."""

    @abc.abstractmethod
    def reset(self, state: G1State) -> None:
        """Called at the start of an episode or when policy is (re)started."""
        pass

    @abc.abstractmethod
    def step(self, state: G1State) -> G1Action:
        """Compute action from current state. Called at policy rate (e.g., 50 Hz)."""
        pass

    @abc.abstractmethod
    def get_observation(self, state: G1State) -> G1Observation:
        """Extract observation from state for logging/debugging."""
        pass

    @property
    @abc.abstractmethod
    def action_dim(self) -> int:
        """Dimension of action space."""
        pass

    @property
    @abc.abstractmethod
    def obs_dim(self) -> int:
        """Dimension of observation space."""
        pass

    def configure(self, config: Dict[str, Any]) -> None:
        """Optional configuration method."""
        pass

    def get_info(self) -> Dict[str, Any]:
        """Return policy info for logging."""
        return {}


class UnitreeG1Robot:
    """
    Main robot class that handles:
    - DDS communication (sim or real)
    - Two-thread architecture (500 Hz control + policy rate)
    - Safety monitoring
    - Sim-to-real deployment
    """

    def __init__(
        self,
        mode: str = 'sim',
        iface: str = 'Ethernet',
        domain_id: Optional[int] = None,
        num_motors: int = NUM_BODY_MOTORS,
        policy_rate: int = CONTROL_FREQ,
        fix_base: bool = True,
        enable_elastic_band: bool = True,
        robot: Optional[G1Robot] = None
    ):
        self.mode = mode.lower()
        self.iface = iface
        self.domain_id = domain_id
        self.num_motors = num_motors
        self.policy_rate = policy_rate
        self.policy_dt = 1.0 / policy_rate
        self.control_dt = 1.0 / PHYSICS_FREQ

        self.bridge: G1Bridge = create_g1_bridge(
            mode=self.mode,
            iface=self.iface,
            domain_id=self.domain_id,
            robot=robot,
            num_motors=self.num_motors
        )

        self._running = False
        self._policy: Optional[Policy] = None

        self._control_thread: Optional[threading.Thread] = None
        self._policy_thread: Optional[threading.Thread] = None

        self._state_lock = threading.Lock()
        self._latest_state: Optional[G1State] = None
        self._latest_action: Optional[G1Action] = None

        self._step_count = 0
        self._start_time = 0.0

        self._safety_limits = {
            'max_joint_vel': 1.2,
            'max_joint_acc': 10.0,
            'max_torque': 100.0,
            'fall_height_threshold': 0.45,
        }

        self._collision_detected = False
        self._emergency_stopped = False

        print(f"[UnitreeG1Robot] Initialized in {self.mode.upper()} mode")
        print(f"  - Motors: {self.num_motors}")
        print(f"  - Policy rate: {self.policy_rate} Hz")
        print(f"  - Control rate: {PHYSICS_FREQ} Hz")

    def set_policy(self, policy: Policy) -> None:
        self._policy = policy
        print(f"[UnitreeG1Robot] Policy set: {policy.__class__.__name__}")

    def start(self, wait_for_robot: bool = True, timeout: float = 5.0) -> bool:
        if self._running:
            print("[UnitreeG1Robot] Already running")
            return True

        if self.mode == 'hardware':
            if wait_for_robot:
                if not self.bridge.wait_for_robot(timeout):
                    print("[UnitreeG1Robot] Failed to connect to robot")
                    return False

            policy_callback = self._policy_step if self._policy else None
            if not self.bridge.start_streaming(policy_callback):
                print("[UnitreeG1Robot] Failed to start hardware streaming")
                return False
        else:
            self._running = True
            self._control_thread = threading.Thread(target=self._control_loop, name="G1ControlLoop", daemon=True)
            self._control_thread.start()

            if self._policy:
                self._policy_thread = threading.Thread(target=self._policy_loop, name="G1PolicyLoop", daemon=True)
                self._policy_thread.start()

        self._start_time = time.time()
        print("[UnitreeG1Robot] Started successfully")
        return True

    def stop(self) -> None:
        if not self._running:
            return

        self._running = False
        self._emergency_stopped = True

        if self.mode == 'hardware':
            self.bridge.close()
        else:
            if self._control_thread:
                self._control_thread.join(timeout=1.0)
            if self._policy_thread:
                self._policy_thread.join(timeout=1.0)

        print("[UnitreeG1Robot] Stopped")

    def emergency_stop(self) -> None:
        print("[UnitreeG1Robot] !!! EMERGENCY STOP !!!")
        self._emergency_stopped = True
        if self.mode == 'hardware':
            self.bridge.emergency_stop()
        self.stop()

    def _control_loop(self) -> None:
        """500 Hz control loop."""
        while self._running and not self._emergency_stopped:
            loop_start = time.time()

            self._update_state()

            if self._latest_action is not None:
                self.bridge.send_joint_targets(
                    q_des=self._latest_action.q_des,
                    kp=self._latest_action.kp,
                    kd=self._latest_action.kd,
                    dq_des=self._latest_action.dq_des,
                    tau_ff=self._latest_action.tau_ff
                )

            self._check_safety()

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, self.control_dt - elapsed)
            time.sleep(sleep_time)

    def _policy_loop(self) -> None:
        """Policy rate loop."""
        while self._running and not self._emergency_stopped:
            loop_start = time.time()

            self._policy_step()

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, self.policy_dt - elapsed)
            time.sleep(sleep_time)

    def _policy_step(self) -> None:
        if self._policy is None or self._latest_state is None:
            return

        try:
            action = self._policy.step(self._latest_state)
            with self._state_lock:
                self._latest_action = action
        except Exception as e:
            print(f"[UnitreeG1Robot] Policy step error: {e}")
            self.emergency_stop()

    def _update_state(self) -> None:
        q, dq = self.bridge.get_joint_states()
        imu_quat, imu_angvel, imu_linacc = self.bridge.get_imu_state()

        state = G1State(
            timestamp=time.time() - self._start_time,
            q=q[:self.num_motors] if len(q) >= self.num_motors else q,
            dq=dq[:self.num_motors] if len(dq) >= self.num_motors else dq,
            imu_quat=imu_quat,
            imu_angvel=imu_angvel,
            imu_linacc=imu_linacc,
        )

        if hasattr(self.bridge, 'robot') and self.bridge.robot is not None:
            robot = self.bridge.robot
            state.base_pos, state.base_quat = robot.get_base_pose()
            state.base_linvel, state.base_angvel = robot.get_base_vel()
            state.left_hand_pos = robot.get_hand_pos(False)
            state.right_hand_pos = robot.get_hand_pos(True)
            state.left_foot_contact, state.right_foot_contact = robot.get_foot_contacts()

        with self._state_lock:
            self._latest_state = state

    def _check_safety(self) -> None:
        if self._latest_state is None:
            return


        if np.any(np.abs(self._latest_state.dq) > self._safety_limits['max_joint_vel']):
            print("[UnitreeG1Robot] Safety: Joint velocity limit exceeded")
            self.emergency_stop()
            return


        if self.mode == 'sim' and hasattr(self.bridge, 'robot'):
            base_pos, _ = self.bridge.robot.get_base_pose()
            if base_pos[2] < self._safety_limits['fall_height_threshold']:
                print(f"[UnitreeG1Robot] Safety: Robot fell (height={base_pos[2]:.2f}m)")
                self.emergency_stop()

    def get_state(self) -> Optional[G1State]:
        with self._state_lock:
            return self._latest_state.copy() if self._latest_state else None

    def get_action(self) -> Optional[G1Action]:
        with self._state_lock:
            return self._latest_action.copy() if self._latest_action else None

    def is_running(self) -> bool:
        return self._running and not self._emergency_stopped

    def is_hardware_active(self) -> bool:
        return self.bridge.is_hardware_active()

    def set_safety_limit(self, key: str, value: float) -> None:
        if key in self._safety_limits:
            self._safety_limits[key] = value

    def get_info(self) -> Dict[str, Any]:
        return {
            'mode': self.mode,
            'running': self._running,
            'hardware_active': self.is_hardware_active(),
            'step_count': self._step_count,
            'uptime': time.time() - self._start_time if self._start_time > 0 else 0,
            'safety_limits': self._safety_limits,
            'emergency_stopped': self._emergency_stopped,
        }


def run_policy(
    policy: Policy,
    mode: str = 'sim',
    iface: str = 'Ethernet',
    domain_id: Optional[int] = None,
    num_motors: int = NUM_BODY_MOTORS,
    policy_rate: int = CONTROL_FREQ,
    duration: Optional[float] = None,
    fix_base: bool = True,
    enable_elastic_band: bool = True,
) -> UnitreeG1Robot:
    """Convenience function to run a policy."""
    robot = UnitreeG1Robot(
        mode=mode,
        iface=iface,
        domain_id=domain_id,
        num_motors=num_motors,
        policy_rate=policy_rate,
        fix_base=fix_base,
        enable_elastic_band=enable_elastic_band,
    )
    robot.set_policy(policy)

    if robot.start():
        try:
            if duration:
                time.sleep(duration)
            else:
                while robot.is_running():
                    time.sleep(1.0)
        except KeyboardInterrupt:
            print("\n[UnitreeG1Robot] Interrupted by user")
        finally:
            robot.stop()
    return robot
