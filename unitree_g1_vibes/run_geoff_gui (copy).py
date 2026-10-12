from __future__ import annotations
import argparse
import sys
import threading
import time
from typing import Any, Tuple
try:
    from PySide6 import QtCore
except ModuleNotFoundError as exc:
    raise SystemExit('PySide6 is required for run_geoff_gui.py – install with:\n    pip install pyside6 pyqtgraph') from exc
from run_geoff_stack import _rx_realsense, _state, _state_lock

def _rx_battery(stop: 'threading.Event', iface: str):
    try:
        import time
        from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize

        def _publish(soc_val: int | None=None, voltage: float | None=None):
            with _state_lock:
                if soc_val is not None:
                    _state['soc'] = soc_val
                if voltage is not None:
                    _state['voltage'] = voltage

        def _attempt_sub(name: str, msg_type, cb):
            try:
                sub = ChannelSubscriber(name, msg_type)
                sub.Init(cb, 50)
                return True
            except Exception:
                return False
        ok = False
        try:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

            def _cb_go(msg: LowState_):
                soc_val = getattr(getattr(msg, 'bms_state', None), 'soc', None)
                if soc_val is not None and soc_val > 0:
                    _publish(int(soc_val))
                else:
                    _publish(voltage=float(msg.power_v))
            ok = _attempt_sub('rt/lowstate', LowState_, _cb_go)
        except Exception:
            ok = False
        if not ok:
            try:
                from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_

                def _cb_hg(msg: BmsState_):
                    _publish(int(msg.soc))
                ok = _attempt_sub('rt/bmsstate', BmsState_, _cb_hg)
            except Exception:
                ok = False
        if not ok:
            try:
                ChannelFactoryInitialize(0, iface)
            except Exception:
                pass
            if not ok:
                ok = _attempt_sub('rt/lowstate', LowState_, _cb_go) if 'LowState_' in locals() else False
                if not ok and 'BmsState_' in locals():
                    ok = _attempt_sub('rt/bmsstate', BmsState_, _cb_hg)
        if not ok:
            raise RuntimeError('Could not subscribe to any battery SOC topic')
        while not stop.is_set():
            time.sleep(0.5)
    except Exception as exc:
        import sys
        print('[run_geoff_gui] Battery monitor disabled:', exc, file=sys.stderr)
_slam_latest: Tuple[Any, Any] | None = None
_slam_lock = threading.Lock()

def _patch_live_slam_for_pyqt() -> None:
    import numpy as np

    class _QtViewer:

        def __init__(self):
            self._latest_pts: np.ndarray | None = None
            self._latest_pose: np.ndarray | None = None

        def push(self, xyz: np.ndarray, pose: np.ndarray):
            global _slam_latest
            with _slam_lock:
                _slam_latest = (xyz, pose)

        def tick(self) -> bool:
            return True

        def close(self):
            pass
    import live_slam as _ls
    _ls._Viewer = _QtViewer

def _run_slam(store_evt: threading.Event):
    try:
        _patch_live_slam_for_pyqt()
        import live_slam as _ls
        demo = _ls.LiveSLAMDemo()
        while not store_evt.is_set():
            time.sleep(0.05)
        demo.shutdown()
    except Exception as exc:
        print('[run_geoff_gui] SLAM thread disabled:', exc, file=sys.stderr)

class GeoffWindow(QtCore.QObject):

    def __init__(self, iface: str, ground_clear_in: float):
        super().__init__()
        from PySide6 import QtWidgets, QtGui
        self._clear_m = ground_clear_in * 0.0254
        import pyqtgraph.opengl as gl
        self.app = QtWidgets.QApplication(sys.argv)
        self.rgb_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.depth_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.rgb_lbl.setMinimumSize(640, 320)
        self.depth_lbl.setMinimumSize(640, 320)
        import pyqtgraph as pg
        self.map_view = pg.GraphicsLayoutWidget()
        self.map_view.setMinimumSize(640, 320)
        self._map_vb = self.map_view.addViewBox(lockAspect=True, enableMouse=True)
        self._map_vb.setMenuEnabled(False)
        self._map_vb.invertY(True)
        self._map_img = pg.ImageItem()
        self._map_vb.addItem(self._map_img)
        self._occ_map: 'np.ndarray | None' = None
        self._map_meta: tuple[float, float, float] | None = None
        self._path_px: list[tuple[int, int]] | None = None
        self.map_view.scene().sigMouseClicked.connect(self._on_map_click)
        self.gl_view = gl.GLViewWidget()
        self.gl_view.opts['distance'] = 30
        self.gl_view.setCameraPosition(distance=30, elevation=20, azimuth=45)
        self.gl_view.setMinimumWidth(640)
        self._scatter = gl.GLScatterPlotItem()
        self.gl_view.addItem(self._scatter)
        self._pose_items: list[gl.GLLinePlotItem] = []
        splitter = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(left)
        v.addWidget(self.rgb_lbl)
        v.addWidget(self.depth_lbl)
        v.addWidget(self.map_view)
        splitter.addWidget(left)
        splitter.addWidget(self.gl_view)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([640, 640])
        self.win = QtWidgets.QMainWindow()
        self.win.setWindowTitle('Geoff-Stack')
        self.win.setCentralWidget(splitter)
        self.win.resize(1600, 760)
        self.status = QtWidgets.QLabel()
        self.win.statusBar().addWidget(self.status)
        self._refresh = QtCore.QTimer()
        self._refresh.setInterval(30)
        self._refresh.timeout.connect(self._on_tick)
        self._refresh.start()
        self._stop_evt = threading.Event()
        self._pressed: set[object] = set()
        self._vx = 0.0
        self._vy = 0.0
        self._omega = 0.0
        self._bal_mode: int = -1
        try:
            from hanger_boot_sequence import hanger_boot_sequence
            self._bot = hanger_boot_sequence(iface=iface)
        except Exception as exc:
            print('[run_geoff_gui] Tele-op disabled:', exc, file=sys.stderr)
            self._bot = None
        self._drive_timer = QtCore.QTimer()
        self._drive_timer.setInterval(100)
        self._drive_timer.timeout.connect(self._on_drive_tick)
        self._drive_timer.start()
        self.app.installEventFilter(self)
        self._threads = [threading.Thread(target=_rx_realsense, args=(self._stop_evt,), daemon=True), threading.Thread(target=_run_slam, args=(self._stop_evt,), daemon=True), threading.Thread(target=_rx_battery, args=(self._stop_evt, iface), daemon=True)]
        for t in self._threads:
            t.start()
        self.app.aboutToQuit.connect(self._on_quit)

    def _numpy_to_qpix(self, bgr):
        import numpy as np
        from PySide6 import QtGui
        if bgr is None or bgr.dtype != np.uint8:
            return None
        h, w, _ = bgr.shape
        qimg = QtGui.QImage(bgr.data.tobytes(), w, h, 3 * w, QtGui.QImage.Format_BGR888)
        return QtGui.QPixmap.fromImage(qimg.copy())

    def _on_tick(self):
        import numpy as np
        with _state_lock:
            rgbd = _state.get('rgbd')
            vx, vy, om = _state.get('vel', (0.0, 0.0, 0.0))
            soc = _state.get('soc')
        if rgbd is not None and rgbd.shape == (480, 1280, 3):
            rgb, depth = (rgbd[:, :640], rgbd[:, 640:])
            px1, px2 = (self._numpy_to_qpix(rgb), self._numpy_to_qpix(depth))
            if px1:
                self.rgb_lbl.setPixmap(px1)
            if px2:
                self.depth_lbl.setPixmap(px2)
        status_txt = f'vx {vx:+.2f}  vy {vy:+.2f}  omega {om:+.2f}'
        if soc is not None:
            status_txt += f'   battery {soc:3d}%'
        else:
            with _state_lock:
                volt = _state.get('voltage')
            if volt is not None:
                status_txt += f'   V {volt:5.1f}'
        self.status.setText(status_txt)
        with _slam_lock:
            data = _slam_latest
        if data is None:
            return
        xyz, pose = data
        if xyz.shape[0] == 0:
            return
        if xyz.shape[0] > 200000:
            xyz = xyz[::int(xyz.shape[0] / 200000) + 1]
        z_ft = xyz[:, 2] * 3.28084
        z_rel = z_ft - z_ft.min()
        _SPAN_FT = 9.0
        v = np.clip(z_rel / _SPAN_FT, 0.0, 1.0)
        _GAMMA = 0.35
        v_gamma = v ** _GAMMA
        try:
            import pyqtgraph as pg
            cmap = pg.colormap.get('turbo')
            colors = cmap.map(v_gamma, mode='float')
        except Exception:
            h = 0.66 * (1.0 - v_gamma)
            s = np.ones_like(h)
            val = np.ones_like(h)
            i = np.floor(h * 6).astype(int)
            f = h * 6 - i
            p = val * (1 - s)
            q = val * (1 - f * s)
            t = val * (1 - (1 - f) * s)
            r = np.choose(i % 6, [val, q, p, p, t, val])
            g = np.choose(i % 6, [t, val, val, q, p, p])
            b = np.choose(i % 6, [p, p, t, val, val, q])
            colors = np.stack([r, g, b, np.ones_like(r)], axis=1)
        self._scatter.setData(pos=xyz, size=1.0, color=colors)
        self._update_2d_map(xyz, pose)
        if pose is not None and pose.shape == (4, 4):
            self._update_pose_axes(pose, xyz)
    _LIN_STEP = 0.05
    _ANG_STEP = 0.2

    @staticmethod
    def _clamp(val: float, limit: float=0.6) -> float:
        return max(-limit, min(limit, val))

    def eventFilter(self, _obj, ev):
        from PySide6 import QtCore
        if ev.type() == QtCore.QEvent.KeyPress:
            if ev.isAutoRepeat():
                return False
            key = ev.key()
            name = self._qt_key_name(key, ev.text())
            if name is not None:
                self._pressed.add(name)
                return True
        elif ev.type() == QtCore.QEvent.KeyRelease:
            if ev.isAutoRepeat():
                return False
            key = ev.key()
            name = self._qt_key_name(key, ev.text())
            if name is not None:
                self._pressed.discard(name)
                return True
        return False

    @staticmethod
    def _qt_key_name(key: int, text: str | None) -> str | None:
        from PySide6 import QtCore
        mapping = {QtCore.Qt.Key_Space: 'space', QtCore.Qt.Key_Escape: 'esc', QtCore.Qt.Key_Z: 'z'}
        if key in mapping:
            return mapping[key]
        if text:
            ch = text.lower()
            if ch in ('w', 'a', 's', 'd', 'q', 'e'):
                return ch
        return None

    def _is_pressed(self, name: str) -> bool:
        return name in self._pressed

    def _on_drive_tick(self):
        if self._is_pressed('w') and (not self._is_pressed('s')):
            self._vx = self._clamp(self._vx + self._LIN_STEP)
        elif self._is_pressed('s') and (not self._is_pressed('w')):
            self._vx = self._clamp(self._vx - self._LIN_STEP)
        else:
            self._vx = 0.0
        if self._is_pressed('q') and (not self._is_pressed('e')):
            self._vy = self._clamp(self._vy + self._LIN_STEP)
        elif self._is_pressed('e') and (not self._is_pressed('q')):
            self._vy = self._clamp(self._vy - self._LIN_STEP)
        else:
            self._vy = 0.0
        if self._is_pressed('a') and (not self._is_pressed('d')):
            self._omega = self._clamp(self._omega + self._ANG_STEP)
        elif self._is_pressed('d') and (not self._is_pressed('a')):
            self._omega = self._clamp(self._omega - self._ANG_STEP)
        else:
            self._omega = 0.0
        if self._is_pressed('space'):
            self._vx = self._vy = self._omega = 0.0
        if self._is_pressed('z'):
            if self._bot is not None:
                try:
                    self._bot.Damp()
                except Exception:
                    pass
            self.app.quit()
            return
        if self._is_pressed('esc'):
            if self._bot is not None:
                try:
                    self._bot.StopMove()
                    self._bot.ZeroTorque()
                except Exception:
                    pass
            self.app.quit()
            return
        if self._bot is not None:
            try:
                self._bot.Move(self._vx, self._vy, self._omega, continous_move=True)
                desired_mode = 0 if self._vx == self._vy == self._omega == 0.0 else 1
                if desired_mode != self._bal_mode:
                    try:
                        self._bot.SetBalanceMode(desired_mode)
                        self._bal_mode = desired_mode
                    except Exception:
                        pass
            except Exception as exc:
                print('[run_geoff_gui] Move failed:', exc, file=sys.stderr)
                self._bot = None
        with _state_lock:
            _state['vel'] = (self._vx, self._vy, self._omega)

    def _on_map_click(self, ev):
        import numpy as np
        if self._occ_map is None or self._map_meta is None:
            return
        pos = ev.scenePos()
        view_pt = self._map_vb.mapSceneToView(pos)
        gx, gy = (int(view_pt.x()), int(view_pt.y()))
        if not (0 <= gx < 480 and 0 <= gy < 480):
            return
        rob_px = getattr(self, '_robot_px', None)
        if rob_px is None:
            return
        rx, ry = rob_px
        if not getattr(ev, 'double', lambda: False)():
            return
        path = self._plan_path(rx, ry, gx, gy, self._occ_map)
        if path is not None and len(path) > 1:
            self._path_px = path
        else:
            print('[run_geoff_gui] No path found to clicked target.')
        self._on_tick()

    @staticmethod
    def _plan_path(sx: int, sy: int, gx: int, gy: int, occ: 'np.ndarray') -> list[tuple[int, int]] | None:
        import heapq
        import math
        import numpy as np
        import cv2
        h, w = occ.shape
        if not (0 <= sx < w and 0 <= sy < h and (0 <= gx < w) and (0 <= gy < h)):
            return None
        if occ[sy, sx] or occ[gy, gx]:
            return None
        free_uint8 = (~occ).astype(np.uint8)
        dist = cv2.distanceTransform(free_uint8, cv2.DIST_L2, 5)
        max_dist = float(dist.max()) or 1.0
        _BIAS = 3.0

        def cell_cost(x: int, y: int) -> float:
            d_norm = dist[y, x] / max_dist
            return 1.0 + _BIAS * (1.0 - d_norm)
        open_set: list[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, (sx, sy)))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score = {(sx, sy): 0.0}

        def heuristic(x: int, y: int) -> float:
            return math.hypot(gx - x, gy - y)
        while open_set:
            _, current = heapq.heappop(open_set)
            cx, cy = current
            if current == (gx, gy):
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == dy == 0:
                        continue
                    nx, ny = (cx + dx, cy + dy)
                    if not (0 <= nx < w and 0 <= ny < h):
                        continue
                    if occ[ny, nx]:
                        continue
                    step = math.hypot(dx, dy) * cell_cost(nx, ny)
                    tentative = g_score[current] + step
                    if tentative < g_score.get((nx, ny), float('inf')):
                        came_from[nx, ny] = current
                        g_score[nx, ny] = tentative
                        f = tentative + heuristic(nx, ny)
                        heapq.heappush(open_set, (f, (nx, ny)))
        return None

    def _on_quit(self):
        self._stop_evt.set()
        self._drive_timer.stop()
        for t in self._threads:
            t.join(timeout=1.0)

    def _update_pose_axes(self, pose: 'np.ndarray', pts: 'np.ndarray') -> None:
        import numpy as np
        import pyqtgraph.opengl as gl
        for item in self._pose_items:
            self.gl_view.removeItem(item)
        self._pose_items.clear()
        size = 0.5
        if pts.shape[0] > 0:
            span = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
            size = max(0.2, min(span * 0.03, 2.0))
        origin = pose[:3, 3]
        rot = pose[:3, :3]
        axes = {(1.0, 0.0, 0.0, 1.0): rot @ np.array([size, 0, 0]), (0.0, 1.0, 0.0, 1.0): rot @ np.array([0, size, 0]), (0.0, 0.0, 1.0, 1.0): rot @ np.array([0, 0, size])}
        for color, vec in axes.items():
            pts_arr = np.vstack([origin, origin + vec])
            item = gl.GLLinePlotItem(pos=pts_arr, color=color, width=2, antialias=True)
            self.gl_view.addItem(item)
            self._pose_items.append(item)

    def _update_2d_map(self, xyz: 'np.ndarray', pose: 'np.ndarray' | None) -> None:
        import numpy as np
        import cv2
        if xyz.shape[0] == 0:
            return
        min_x, max_x = (float(xyz[:, 0].min()), float(xyz[:, 0].max()))
        min_y, max_y = (float(xyz[:, 1].min()), float(xyz[:, 1].max()))
        span = max(max_x - min_x, max_y - min_y, 1e-06)
        scale = 470.0 / span
        self._map_meta = (min_x, min_y, scale)

        def world_to_px(xw: 'np.ndarray', yw: 'np.ndarray') -> tuple['np.ndarray', 'np.ndarray']:
            px = ((yw - min_y) * scale + 5).astype(np.int32)
            py = ((xw - min_x) * scale + 5).astype(np.int32)
            py = 479 - py
            return (px, py)
        canvas = np.full((480, 480, 3), 30, dtype=np.uint8)
        ground_z_inst = float(np.percentile(xyz[:, 2], 5.0))
        _ALPHA = 0.05
        if not hasattr(self, '_ground_z_smooth'):
            self._ground_z_smooth = ground_z_inst
        else:
            self._ground_z_smooth = (1.0 - _ALPHA) * self._ground_z_smooth + _ALPHA * ground_z_inst
        ground_z = float(self._ground_z_smooth)
        import os as _os
        try:
            _R_XY = float(_os.environ.get('LIDAR_SELF_FILTER_RADIUS', 0.3))
            _DZ = float(_os.environ.get('LIDAR_SELF_FILTER_Z', 0.24))
        except ValueError:
            _R_XY, _DZ = (0.08, 0.05)
        if pose is not None and pose.shape == (4, 4):
            rob_pos = pose[:3, 3]
            diff = xyz - rob_pos
            dist_xy = np.linalg.norm(diff[:, :2], axis=1)
            close = dist_xy < _R_XY
            near_plane = np.abs(diff[:, 2]) < _DZ
            keep_mask = ~(close & near_plane)
            if keep_mask.sum() != xyz.shape[0]:
                xyz = xyz[keep_mask]
        thresh = ground_z + self._clear_m
        pts = xyz[xyz[:, 2] > thresh]
        occ = np.zeros((480, 480), dtype=bool)
        if pts.shape[0] > 0:
            x_obs, y_obs = (pts[:, 0], pts[:, 1])
            px_obs, py_obs = world_to_px(x_obs, y_obs)
            valid = (px_obs >= 0) & (px_obs < 480) & (py_obs >= 0) & (py_obs < 480)
            px_obs, py_obs = (px_obs[valid], py_obs[valid])
            occ[py_obs, px_obs] = True
            canvas[py_obs, px_obs] = (255, 255, 255)
        cv2.rectangle(canvas, (0, 0), (479, 479), (255, 255, 255), 1)
        if pose is not None and pose.shape == (4, 4):
            rob_pos = pose[:3, 3]
            rx, ry = world_to_px(np.array([rob_pos[0]]), np.array([rob_pos[1]]))
            rx, ry = (int(rx[0]), int(ry[0]))
            self._robot_px = (rx, ry)
            rr0, rr1 = (max(0, ry - 1), min(480, ry + 2))
            rc0, rc1 = (max(0, rx - 1), min(480, rx + 2))
            occ[rr0:rr1, rc0:rc1] = False
            fwd_m = 0.25
            fwd_vec = pose[:3, 0] * fwd_m
            tip_world = rob_pos + fwd_vec
            tx, ty = world_to_px(np.array([tip_world[0]]), np.array([tip_world[1]]))
            tx, ty = (int(tx[0]), int(ty[0]))
            cv2.arrowedLine(canvas, (rx, ry), (tx, ty), (0, 255, 0), 2, tipLength=0.8)
        if self._path_px and len(self._path_px) > 1:
            cv2.polylines(canvas, [np.array(self._path_px, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)
            gx, gy = self._path_px[-1]
            cv2.circle(canvas, (gx, gy), 4, (0, 0, 255), -1)
        self._occ_map = occ.copy()
        try:
            self._map_img.setImage(canvas, levels=(0, 255))
        except Exception:
            px = self._numpy_to_qpix(canvas)
            if px and hasattr(self, 'rgb_lbl'):
                from PySide6 import QtWidgets as _QtW
                if not hasattr(self, '_legacy_lbl'):
                    self._legacy_lbl = _QtW.QLabel(alignment=QtCore.Qt.AlignCenter)
                    self._legacy_lbl.setMinimumSize(640, 320)
                    self._map_vb.hide()
                    self.map_view.setParent(None)
                    self.rgb_lbl.parentWidget().layout().addWidget(self._legacy_lbl)
                self._legacy_lbl.setPixmap(px)

    def run(self):
        self.win.show()
        sys.exit(self.app.exec())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--iface', default='enp68s0f1', help='NIC connected to the Unitree G-1')
    parser.add_argument('--clear', type=float, default=18.0, help='Clearance (in inches) above detected floor before a point is tagged as an obstacle')
    args = parser.parse_args()
    GeoffWindow(args.iface, args.clear).run()
if __name__ == '__main__':
    main()
