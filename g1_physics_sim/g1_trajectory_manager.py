from __future__ import annotations
import json
import math
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
try:  # package and flat-script imports both work
    from .g1_constants import (    JOINT_NAMES_29DOF, NOMINAL_STAND_Q, NUM_BODY_MOTORS,
    L_HIP_PITCH, L_KNEE, L_ANKLE_PITCH, R_HIP_PITCH, R_KNEE, R_ANKLE_PITCH,
    L_SHOULDER_PITCH, L_ELBOW, L_WRIST_ROLL,
    R_SHOULDER_PITCH, R_ELBOW, R_WRIST_ROLL)
except ImportError:  # pragma: no cover - flat script layout
    from g1_constants import (JOINT_NAMES_29DOF, NOMINAL_STAND_Q, NUM_BODY_MOTORS,
L_HIP_PITCH, L_KNEE, L_ANKLE_PITCH, R_HIP_PITCH, R_KNEE, R_ANKLE_PITCH,
L_SHOULDER_PITCH, L_ELBOW, L_WRIST_ROLL,
R_SHOULDER_PITCH, R_ELBOW, R_WRIST_ROLL)
try:  # package and flat-script imports both work
    from .g1_state import (    G1State,
    G1Action)
except ImportError:  # pragma: no cover - flat script layout
    from g1_state import (G1State,
G1Action)


class TrajectoryRecorder:

    def __init__(self, output_dir: Optional[str] = None, num_motors: int = NUM_BODY_MOTORS):
        if num_motors != NUM_BODY_MOTORS:
            raise ValueError('G1 body trajectories are exactly 29 motors')
        if output_dir is None:
            self.output_dir = pathlib.Path(__file__).resolve().parent / 'trajectories'
        else:
            self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_motors = NUM_BODY_MOTORS
        self.joint_names = list(JOINT_NAMES_29DOF)
        self.is_recording = False
        self.frames: List[Dict[str, Any]] = []
        self.start_time = 0.0

    def start(self):
        self.frames.clear()
        self.start_time = time.time()
        self.is_recording = True
        print('\n[RECORDER] >>> Recording STARTED at 50 Hz <<<')

    def record_frame(
        self,
        state: G1State,
        action: Optional[G1Action] = None,
        target_pos: Optional[np.ndarray] = None
    ):
        if not self.is_recording:
            return
        t = time.time() - self.start_time
        frame = {
            'time': round(float(t), 4),
            'joint_q': [round(float(val), 5) for val in state.q[:self.num_motors]],
            'joint_dq': [round(float(val), 4) for val in state.dq[:self.num_motors]],
            'joint_tau': [round(float(val), 4) for val in state.tau_est[:self.num_motors]],
            'base_pos': [round(float(v), 4) for v in state.base_pos],
            'base_quat': [round(float(v), 4) for v in state.base_quat],
            'imu_angvel': [round(float(v), 4) for v in state.imu_angvel],
            'imu_linacc': [round(float(v), 4) for v in state.imu_linacc],
            'left_foot_contact': state.left_foot_contact,
            'right_foot_contact': state.right_foot_contact,
            'left_wrist_pos': [round(float(v), 4) for v in state.left_wrist_pos],
            'right_wrist_pos': [round(float(v), 4) for v in state.right_wrist_pos],
            'mode_pr': state.mode_pr,
            'mode_machine': state.mode_machine,
        }
        if action is not None:
            frame['q_des'] = [round(float(val), 5) for val in action.q_des[:self.num_motors]]
            frame['kp'] = [round(float(val), 2) for val in action.kp[:self.num_motors]]
            frame['kd'] = [round(float(val), 2) for val in action.kd[:self.num_motors]]
        if target_pos is not None:
            frame['target_pos'] = [round(float(v), 4) for v in target_pos]
        self.frames.append(frame)

    def record_state(self, state: G1State, action: Optional[G1Action] = None, target_pos: Optional[np.ndarray] = None):
        self.record_frame(state, action, target_pos)

    def stop_and_save(self, filename: Optional[str] = None) -> Optional[pathlib.Path]:
        if not self.is_recording or len(self.frames) == 0:
            self.is_recording = False
            print('[RECORDER] No frames recorded.')
            return None
        self.is_recording = False
        duration = time.time() - self.start_time
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'motion_{timestamp}.json'
        save_path = self.output_dir / filename
        data = {
            'name': save_path.stem,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'num_motors': self.num_motors,
            'num_frames': len(self.frames),
            'duration_sec': round(duration, 3),
            'fps': round(len(self.frames) / max(0.01, duration), 1),
            'joint_names': self.joint_names,
            'frames': self.frames
        }
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f'\n[RECORDER] >>> Motion SAVED: {save_path.name} ({len(self.frames)} frames, {duration:.1f}s) <<<')
        return save_path


class TrajectoryPlayer:

    def __init__(self, file_path: str | pathlib.Path):
        self.file_path = pathlib.Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f'Trajectory file not found: {self.file_path}')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.frames = self.data.get('frames', [])
        if len(self.frames) == 0:
            raise ValueError('Trajectory contains 0 frames!')

        recorded = int(self.data.get('num_motors', NUM_BODY_MOTORS))
        if recorded != NUM_BODY_MOTORS:
            raise ValueError(
                f'Trajectory has {recorded} motors; G1 body trajectories must be {NUM_BODY_MOTORS}. '
                'Hands are recorded on a separate channel.'
            )
        width = len(self.frames[0].get('joint_q', []))
        if width != NUM_BODY_MOTORS:
            raise ValueError(f'Frame width {width} != {NUM_BODY_MOTORS}')

        self.duration = float(self.data.get('duration_sec', 1.0))
        if self.duration <= 0:
            self.duration = float(self.times[-1]) if self.times[-1] > 0 else 1.0
        self.num_frames = len(self.frames)
        self.times = np.array([f['time'] for f in self.frames], dtype=np.float64)
        if not np.all(np.diff(self.times) >= 0):
            self.times = np.sort(self.times)
        self.joint_trajectories = np.array([f['joint_q'] for f in self.frames], dtype=np.float32)
        self.q_des_trajectories = None
        if 'q_des' in self.frames[0]:
            self.q_des_trajectories = np.array([f['q_des'] for f in self.frames], dtype=np.float32)
        self.kp_trajectories = None
        if 'kp' in self.frames[0]:
            self.kp_trajectories = np.array([f['kp'] for f in self.frames], dtype=np.float32)
        self.kd_trajectories = None
        if 'kd' in self.frames[0]:
            self.kd_trajectories = np.array([f['kd'] for f in self.frames], dtype=np.float32)
        self.is_playing = False
        self.play_start_time = 0.0
        self.loop = False

    def start_playback(self, loop: bool = False):
        self.loop = loop
        self.play_start_time = time.time()
        self.is_playing = True
        print(f"\n[PLAYER] >>> Playing trajectory '{self.file_path.name}' ({self.duration:.1f}s, Loop={loop}) <<<")

    def get_frame(self, speed_mult: float = 1.0):
        """Interpolate the trajectory at the current wall-clock offset."""
        if not self.is_playing:
            return (None, None, None, None, True)
        elapsed = (time.time() - self.play_start_time) * max(0.0, speed_mult)
        is_finished = False
        if elapsed >= self.duration:
            if self.loop:
                elapsed = elapsed % max(1e-3, self.duration)
            else:
                elapsed = self.duration
                is_finished = True
                self.is_playing = False

        def interp(mat: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if mat is None:
                return None
            return np.array([np.interp(elapsed, self.times, mat[:, i])
                             for i in range(mat.shape[1])], dtype=np.float32)

        return (interp(self.joint_trajectories), interp(self.q_des_trajectories),
                interp(self.kp_trajectories), interp(self.kd_trajectories), is_finished)

    def get_action(self, speed_mult: float = 1.0) -> Tuple[Optional[G1Action], bool]:
        """Measured-joint playback action with official default gains."""
        q, q_des, kp, kd, is_finished = self.get_frame(speed_mult)
        if q is None:
            return (None, True)
        action = G1Action()
        # Trajectories store the measured joint_q; when a q_des channel exists
        # it is the authoritative command (records both).
        action.q_des[:] = q_des if q_des is not None else q
        action.kp[:] = kp if kp is not None else OFFICIAL_KP
        action.kd[:] = kd if kd is not None else OFFICIAL_KD
        return (action, is_finished)

    def stop(self):
        self.is_playing = False


class G1MovementScripts:

    @staticmethod
    def create_wave_script(duration: float = 4.0, fps: int = 50, right_arm: bool = False) -> Dict[str, Any]:
        num_frames = int(duration * fps)
        frames = []
        base_q = NOMINAL_STAND_Q.copy()
        pitch_idx = R_SHOULDER_PITCH if right_arm else L_SHOULDER_PITCH
        elbow_idx = R_ELBOW if right_arm else L_ELBOW
        wrist_roll_idx = R_WRIST_ROLL if right_arm else L_WRIST_ROLL
        side = -1.0 if right_arm else 1.0
        for idx in range(num_frames):
            t = idx / fps
            phase = t / duration
            q = base_q.copy()
            q[pitch_idx] = -0.6
            q[pitch_idx + 1] = side * 0.7
            q[elbow_idx] = 1.2
            wave_angle = 0.45 * math.sin(2.0 * math.pi * 2.5 * t)
            q[wrist_roll_idx] = wave_angle
            q[pitch_idx + 2] = side * (0.2 + 0.15 * math.cos(2.0 * math.pi * 2.5 * t))
            frames.append({'time': round(t, 4), 'joint_q': [round(float(v), 5) for v in q]})
        return {
            'name': f"wave_gesture_{('right' if right_arm else 'left')}",
            'duration_sec': duration,
            'fps': fps,
            'num_frames': num_frames,
            'joint_names': JOINT_NAMES_29DOF,
            'frames': frames
        }

    @staticmethod
    def create_circle_cartesian_script(radius: float = 0.08, duration: float = 4.0, fps: int = 50, right_arm: bool = False) -> List[np.ndarray]:
        num_pts = int(duration * fps)
        pts = []
        center_x = 0.22
        center_y = -0.18 if right_arm else 0.18
        center_z = 0.88
        for i in range(num_pts):
            angle = 2.0 * math.pi * (i / num_pts)
            y = center_y + radius * math.cos(angle)
            z = center_z + radius * math.sin(angle)
            pts.append(np.array([center_x, y, z], dtype=np.float32))
        return pts

    @staticmethod
    def create_square_cartesian_script(size: float = 0.12, duration: float = 4.0, fps: int = 50, right_arm: bool = False) -> List[np.ndarray]:
        num_pts = int(duration * fps)
        pts = []
        center_x = 0.22
        cy = -0.18 if right_arm else 0.18
        cz = 0.88
        half = size / 2.0
        corners = [
            np.array([center_x, cy - half, cz - half]),
            np.array([center_x, cy + half, cz - half]),
            np.array([center_x, cy + half, cz + half]),
            np.array([center_x, cy - half, cz + half])
        ]
        pts_per_side = num_pts // 4
        for side_idx in range(4):
            p_start = corners[side_idx]
            p_end = corners[(side_idx + 1) % 4]
            for step in range(pts_per_side):
                alpha = step / pts_per_side
                pts.append(p_start + alpha * (p_end - p_start))
        return pts

    @staticmethod
    def create_standup_script(duration: float = 3.0, fps: int = 50) -> Dict[str, Any]:
        """Script to stand up from zero pose to nominal stand."""
        num_frames = int(duration * fps)
        frames = []
        for idx in range(num_frames):
            t = idx / fps
            ratio = t / duration
            q = (1.0 - ratio) * np.zeros(NUM_BODY_MOTORS) + ratio * NOMINAL_STAND_Q
            frames.append({'time': round(t, 4), 'joint_q': [round(float(v), 5) for v in q]})
        return {
            'name': 'standup_from_zero',
            'duration_sec': duration,
            'fps': fps,
            'num_frames': num_frames,
            'joint_names': JOINT_NAMES_29DOF,
            'frames': frames
        }

    @staticmethod
    def create_squat_script(duration: float = 2.0, fps: int = 50, depth: float = 0.5) -> Dict[str, Any]:
        """Script to squat from standing."""
        num_frames = int(duration * fps)
        frames = []
        stand_q = NOMINAL_STAND_Q.copy()
        squat_q = stand_q.copy()
        squat_q[L_HIP_PITCH] = -0.8
        squat_q[L_KNEE] = 1.5
        squat_q[L_ANKLE_PITCH] = -0.7
        squat_q[R_HIP_PITCH] = -0.8
        squat_q[R_KNEE] = 1.5
        squat_q[R_ANKLE_PITCH] = -0.7
        for idx in range(num_frames):
            t = idx / fps
            ratio = t / duration
            q = (1.0 - ratio) * stand_q + ratio * squat_q
            frames.append({'time': round(t, 4), 'joint_q': [round(float(v), 5) for v in q]})
        return {
            'name': f'squat_depth_{depth}',
            'duration_sec': duration,
            'fps': fps,
            'num_frames': num_frames,
            'joint_names': JOINT_NAMES_29DOF,
            'frames': frames
        }
