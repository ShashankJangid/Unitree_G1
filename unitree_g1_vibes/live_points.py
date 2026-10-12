from __future__ import annotations
import signal
import time
from pathlib import Path
from typing import Optional
import os
MOUNT = os.environ.get('LIVOX_MOUNT', 'upside_down').lower()
if MOUNT not in {'normal', 'upside_down'}:
    raise SystemExit("LIVOX_MOUNT must be 'normal' or 'upside_down'")
import numpy as np
import open3d as o3d
try:
    from livox2_python import Livox2 as _Livox
    _SDK2 = True
except Exception as _e:
    print('[INFO] livox2_python unavailable (', _e, ') – falling back to SDK1.')
    from livox_python import Livox as _Livox
    _SDK2 = False

class _Viewer:

    def __init__(self):
        self._vis = o3d.visualization.Visualizer()
        self._vis.create_window(window_name='Livox – live point-cloud', width=1280, height=720)
        self._frames: list[np.ndarray] = []
        self._max_frames = 15
        self._pcd = o3d.geometry.PointCloud()
        self._vis.add_geometry(self._pcd)
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
        if MOUNT == 'upside_down':
            R180 = np.diag([1.0, -1.0, -1.0, 1.0])
            origin_frame.transform(R180)
        self._vis.add_geometry(origin_frame)
        self._first = True

    def push(self, xyz: np.ndarray):
        self._frames.append(xyz)
        if len(self._frames) > self._max_frames:
            self._frames.pop(0)

    def tick(self) -> bool:
        if self._frames:
            merged = np.concatenate(self._frames, axis=0)
            self._pcd.points = o3d.utility.Vector3dVector(merged)
            self._vis.update_geometry(self._pcd)
            if self._first:
                self._vis.reset_view_point(True)
                self._first = False
            self._latest = None
        alive = self._vis.poll_events()
        self._vis.update_renderer()
        return alive

    def close(self):
        self._vis.destroy_window()

class LiveViewer(_Livox):

    def __init__(self):
        if _SDK2:
            super().__init__('mid360_config.json', host_ip='192.168.123.222')
        else:
            super().__init__()
        self._view = _Viewer()

    def handle_points(self, xyz: np.ndarray):
        if MOUNT == 'upside_down':
            xyz = xyz * np.array([1.0, -1.0, -1.0], dtype=xyz.dtype)
        if xyz.shape[0] > 100000:
            step = xyz.shape[0] // 100000
            xyz = xyz[::step]
        self._view.push(xyz)

    def shutdown(self):
        super().shutdown()
        self._view.close()

def main() -> None:
    lidar = LiveViewer()
    stop = False

    def _sigint(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGINT, _sigint)
    try:
        while not stop and lidar._view.tick():
            time.sleep(0.01)
    finally:
        lidar.shutdown()
if __name__ == '__main__':
    main()
