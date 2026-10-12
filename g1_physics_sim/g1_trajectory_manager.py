from __future__ import annotations
import json
import math
import pathlib
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from g1_constants import JOINT_NAMES_29DOF, NOMINAL_STAND_Q, NUM_MOTORS, L_SHOULDER_PITCH, L_ELBOW, L_WRIST_ROLL, R_SHOULDER_PITCH, R_ELBOW, R_WRIST_ROLL

class TrajectoryRecorder:

    def __init__(self, output_dir: Optional[str]=None):
        if output_dir is None:
            self.output_dir = pathlib.Path(__file__).resolve().parent / 'trajectories'
        else:
            self.output_dir = pathlib.Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.is_recording = False
        self.frames: List[Dict[str, Any]] = []
        self.start_time = 0.0

    def start(self):
        self.frames.clear()
        self.start_time = time.time()
        self.is_recording = True
        print('\n[RECORDER] >>> Recording STARTED at 50 Hz <<<')

    def record_frame(self, q_act: np.ndarray, dq_act: Optional[np.ndarray]=None, target_pos: Optional[np.ndarray]=None, hand_pos_l: Optional[np.ndarray]=None, hand_pos_r: Optional[np.ndarray]=None):
        if not self.is_recording:
            return
        t = time.time() - self.start_time
        frame = {'time': round(float(t), 4), 'joint_q': [round(float(val), 5) for val in q_act[:NUM_MOTORS]], 'joint_dq': [round(float(val), 4) for val in dq_act[:NUM_MOTORS]] if dq_act is not None else [], 'target_pos': [round(float(v), 4) for v in target_pos] if target_pos is not None else [], 'hand_pos_l': [round(float(v), 4) for v in hand_pos_l] if hand_pos_l is not None else [], 'hand_pos_r': [round(float(v), 4) for v in hand_pos_r] if hand_pos_r is not None else []}
        self.frames.append(frame)

    def stop_and_save(self, filename: Optional[str]=None) -> Optional[pathlib.Path]:
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
        data = {'name': save_path.stem, 'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'num_motors': NUM_MOTORS, 'num_frames': len(self.frames), 'duration_sec': round(duration, 3), 'fps': round(len(self.frames) / max(0.01, duration), 1), 'joint_names': JOINT_NAMES_29DOF, 'frames': self.frames}
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
        self.duration = float(self.data.get('duration_sec', 1.0))
        self.num_frames = len(self.frames)
        self.times = np.array([f['time'] for f in self.frames], dtype=np.float32)
        self.joint_trajectories = np.array([f['joint_q'] for f in self.frames], dtype=np.float32)
        self.is_playing = False
        self.play_start_time = 0.0
        self.loop = False

    def start_playback(self, loop: bool=False):
        self.loop = loop
        self.play_start_time = time.time()
        self.is_playing = True
        print(f"\n[PLAYER] >>> Playing trajectory '{self.file_path.name}' ({self.duration:.1f}s, Loop={loop}) <<<")

    def get_frame(self, speed_mult: float=1.0) -> Tuple[Optional[np.ndarray], bool]:
        if not self.is_playing:
            return (None, True)
        elapsed = (time.time() - self.play_start_time) * speed_mult
        is_finished = False
        if elapsed >= self.duration:
            if self.loop:
                elapsed = elapsed % max(0.001, self.duration)
            else:
                elapsed = self.duration
                is_finished = True
                self.is_playing = False
        q_interp = np.zeros(NUM_MOTORS, dtype=np.float32)
        for i in range(NUM_MOTORS):
            q_interp[i] = float(np.interp(elapsed, self.times, self.joint_trajectories[:, i]))
        return (q_interp, is_finished)

    def stop(self):
        self.is_playing = False

class G1MovementScripts:

    @staticmethod
    def create_wave_script(duration: float=4.0, fps: int=50, right_arm: bool=False) -> Dict[str, Any]:
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
        return {'name': f"wave_gesture_{('right' if right_arm else 'left')}", 'duration_sec': duration, 'fps': fps, 'num_frames': num_frames, 'frames': frames}

    @staticmethod
    def create_circle_cartesian_script(radius: float=0.08, duration: float=4.0, fps: int=50, right_arm: bool=False) -> List[np.ndarray]:
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
    def create_square_cartesian_script(size: float=0.12, duration: float=4.0, fps: int=50, right_arm: bool=False) -> List[np.ndarray]:
        num_pts = int(duration * fps)
        pts = []
        center_x = 0.22
        cy = -0.18 if right_arm else 0.18
        cz = 0.88
        half = size / 2.0
        corners = [np.array([center_x, cy - half, cz - half]), np.array([center_x, cy + half, cz - half]), np.array([center_x, cy + half, cz + half]), np.array([center_x, cy - half, cz + half])]
        pts_per_side = num_pts // 4
        for side_idx in range(4):
            p_start = corners[side_idx]
            p_end = corners[(side_idx + 1) % 4]
            for step in range(pts_per_side):
                alpha = step / pts_per_side
                pts.append(p_start + alpha * (p_end - p_start))
        return pts
