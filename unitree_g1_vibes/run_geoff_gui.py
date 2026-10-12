from __future__ import annotations
import argparse
import sys
import threading
import time
from typing import Any, Tuple
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

def _setup_logging() -> logging.Logger:
    log_dir = Path(__file__).resolve().parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / 'run_geoff_gui.log'
    fh = RotatingFileHandler(log_path, mode='w', maxBytes=2000000, backupCount=2, encoding='utf-8')
    _orig_stderr = sys.stderr
    ch = logging.StreamHandler(_orig_stderr)
    fmt = '%(asctime)s  %(levelname)-8s  %(message)s'
    for h in (fh, ch):
        h.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)
    import warnings
    warnings.filterwarnings('ignore', message='A builtin ctypes object gave a PEP3118 format string that does not match its itemsize.*', category=RuntimeWarning, module='numpy\\.ctypeslib')

    class _StreamToLogger:

        def __init__(self, level: int):
            self._level = level
            self._logger = logging.getLogger('geoff_gui')

        def write(self, msg: str):
            msg = msg.rstrip()
            if msg:
                self._logger.log(self._level, msg)

        def flush(self):
            pass
    sys.stdout = _StreamToLogger(logging.INFO)
    sys.stderr = _StreamToLogger(logging.ERROR)
    return logging.getLogger('geoff_gui')
log = _setup_logging()
try:
    from PySide6 import QtCore
except ModuleNotFoundError as exc:
    log.error("PySide6 is required – install via 'pip install pyside6 pyqtgraph'")
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
    try:
        _orig_hp = _ls.LiveSLAMDemo.handle_points

        def _safe_hp(self, xyz):
            try:
                _orig_hp(self, xyz)
            except Exception as exc:
                try:
                    self._viewer.push(xyz, None)
                except Exception:
                    pass
                print('[run_geoff_gui] KISS-ICP first frame failed:', exc)
        _ls.LiveSLAMDemo.handle_points = _safe_hp
    except Exception:
        pass

def _run_slam(stop_evt: threading.Event):
    try:
        _patch_live_slam_for_pyqt()
        import live_slam as _ls
        demo = _ls.LiveSLAMDemo()
        spin_fn = getattr(demo, 'spin', None)
        if callable(spin_fn):
            t_spin = threading.Thread(target=spin_fn, daemon=True)
            t_spin.start()
        try:
            while not stop_evt.is_set():
                try:
                    demo._viewer.tick()
                except Exception:
                    pass
                time.sleep(0.05)
        finally:
            try:
                demo.shutdown()
            except Exception:
                pass
    except Exception as exc:
        print('[run_geoff_gui] SLAM thread disabled:', exc, file=sys.stderr)

class GeoffWindow(QtCore.QObject):

    def __init__(self, iface: str, ground_clear_in: float, *, hand: str='left', grip_force: float | None=None):
        super().__init__()
        from PySide6 import QtWidgets, QtGui
        self._clear_m = ground_clear_in * 0.0254
        self._cli_grip_force = grip_force if grip_force is not None else 0.3
        import pyqtgraph.opengl as gl
        self.app = QtWidgets.QApplication(sys.argv)
        import signal
        try:
            signal.signal(signal.SIGINT, lambda *_: self.app.quit())
        except Exception:
            pass
        self.rgb_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.depth_lbl = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.rgb_lbl.setMinimumSize(640, 320)
        self.depth_lbl.setMinimumSize(640, 320)
        for _lbl in (self.rgb_lbl, self.depth_lbl):
            _lbl.setStyleSheet('background-color: black')
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
        self._btn_damp = QtWidgets.QPushButton('Damp arms & center waist')
        self._btn_damp.setToolTip('Switch upper body to passive mode and set waist to 0 rad')
        self._btn_damp.clicked.connect(self._on_damp_pressed)
        self.win.statusBar().addPermanentWidget(self._btn_damp)
        self._arm_selector = QtWidgets.QComboBox()
        self._arm_selector.addItems(['Left arm', 'Right arm'])
        self._arm_selector.setCurrentIndex(0)
        self._arm_selector.setToolTip('Select which arm to control and run inference on')
        self._active_arm: str = 'left'

        def _on_sel_changed(idx: int):
            self._active_arm = 'left' if idx == 0 else 'right'
            try:
                self._configure_arm_variables()
            except Exception as exc:
                print('[run_geoff_gui] Reconfigure arm failed:', exc, file=sys.stderr)
        self._arm_selector.currentIndexChanged.connect(_on_sel_changed)
        self.win.statusBar().addPermanentWidget(self._arm_selector)
        self._key_overlay = QtWidgets.QWidget(self.gl_view)
        self._key_overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._key_overlay.move(10, 10)
        self._key_overlay.setStyleSheet('background-color: rgba(0, 0, 0, 150);border-radius: 6px;')
        _lay = QtWidgets.QVBoxLayout(self._key_overlay)
        _lay.setContentsMargins(8, 6, 8, 6)
        _lay.setSpacing(0)
        self._header_lbl = QtWidgets.QLabel('Key inputs:')
        self._header_lbl.setStyleSheet("color: #ffff00; font: 12pt 'Consolas', 'Monaco', 'Courier New', monospace;")
        _lay.addWidget(self._header_lbl)
        self._keys_lbl = QtWidgets.QLabel('–')
        self._keys_lbl.setStyleSheet("color: #ffff00; font: bold 24pt 'Consolas', 'Monaco', 'Courier New', monospace;")
        _lay.addWidget(self._keys_lbl)
        self._key_overlay.adjustSize()
        self._key_overlay.show()
        self._keys_opacity = QtWidgets.QGraphicsOpacityEffect(self._keys_lbl)
        self._keys_lbl.setGraphicsEffect(self._keys_opacity)
        self._fade_anim = QtCore.QPropertyAnimation(self._keys_opacity, b'opacity', self)
        self._fade_anim.setDuration(600)

        def _on_fade_finished():
            self._keys_opacity.setOpacity(1.0)
            self._keys_lbl.setText('–')
            self._key_overlay.adjustSize()
        self._fade_anim.finished.connect(_on_fade_finished)
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
        self._arm_pub = None
        try:
            from unitree_sdk2py.core.channel import ChannelPublisher
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.utils.crc import CRC
            _WAIST_YAW_IDX = 12
            _LEFT_IDX = {idx: 0 for idx in range(15, 22)}
            _RIGHT_IDX = {idx: 0 for idx in range(22, 29)}
            _ARM_IDX = _LEFT_IDX if self._active_arm == 'left' else _RIGHT_IDX
            self._arm_joint_idx: list[int] = list(_ARM_IDX.keys())
            self._waist_idx: int = _WAIST_YAW_IDX
            _NOT_USED_IDX = 29
            self._crc = CRC()
            self._arm_cmd = unitree_hg_msg_dds__LowCmd_()
            self._arm_cmd.motor_cmd[_NOT_USED_IDX].q = 1
            if self._active_arm == 'right':
                self._pose_seq: list[list[tuple[int, float]]] = [[(_WAIST_YAW_IDX, 0.0), (22, -0.023), (23, -0.225), (24, +0.502), (25, +1.317), (26, +0.185), (27, +0.125), (28, -0.182)], [(_WAIST_YAW_IDX, 0.0), (22, +0.087), (23, -0.271), (24, +0.323), (25, +0.691), (26, +0.24), (27, -0.771), (28, -0.176)]]
            else:
                self._pose_seq = [[(_WAIST_YAW_IDX, 0.0), (15, +0.211), (16, +0.181), (17, -0.284), (18, +0.672), (19, -0.379), (20, -0.852), (21, -0.019)]]
            self._cmd_q: dict[int, float] = {idx: 0.0 for idx in _ARM_IDX}
            self._cmd_q[_WAIST_YAW_IDX] = 0.0
            self._joint_cur: dict[int, float] = {}
            self._ls_sub = None

            def _init_ls_sub():
                from unitree_sdk2py.core.channel import ChannelSubscriber
                _candidates = ['unitree_sdk2py.idl.unitree_hg.msg.dds_.LowState_', 'unitree_sdk2py.idl.unitree_go.msg.dds_.LowState_']
                for dotted in _candidates:
                    try:
                        mod_path, cls_name = dotted.rsplit('.', 1)
                        mod = __import__(mod_path, fromlist=[cls_name])
                        LowState_ = getattr(mod, cls_name)

                        def _ls_cb(msg):
                            for j_idx in (*self._arm_joint_idx, _WAIST_YAW_IDX):
                                try:
                                    self._joint_cur[j_idx] = msg.motor_state[j_idx].q
                                except Exception:
                                    pass
                        sub = ChannelSubscriber('rt/lowstate', LowState_)
                        sub.Init(_ls_cb, 200)
                        self._ls_sub = sub
                        return
                    except Exception:
                        continue
            threading.Thread(target=_init_ls_sub, daemon=True).start()
            self._initialised_from_state = False
            self._seq_idx = 0
            self._SEQ_EPS = 0.01
            self._STEP = 0.02
            self._arm_pub = ChannelPublisher('rt/arm_sdk', LowCmd_)
            self._arm_pub.Init()
            self._arm_timer = QtCore.QTimer()
            self._arm_timer.setInterval(20)
            self._arm_timer.timeout.connect(self._on_arm_tick)
            self._arm_timer.start()

            def _apply_pose_once(pose: list[tuple[int, float]]):
                for j_idx, q_val in pose:
                    mc = self._arm_cmd.motor_cmd[j_idx]
                    mc.q = q_val
                    mc.dq = 0.0
                    mc.tau = 0.0
                    mc.kp = 60.0
                    mc.kd = 1.5
            if self._active_arm == 'left':
                _ready_other = [(22, +0.087), (23, -0.271), (24, +0.323), (25, +0.691), (26, +0.24), (27, -0.771), (28, -0.176)]
            else:
                _ready_other = [(15, +0.211), (16, +0.181), (17, -0.284), (18, +0.672), (19, -0.379), (20, -0.852), (21, -0.019)]
            _apply_pose_once(_ready_other)
            self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)
            try:
                self._arm_pub.Write(self._arm_cmd)
            except Exception:
                pass
        except Exception as exc:
            print('[run_geoff_gui] Arm control disabled:', exc, file=sys.stderr)
        self._dex3 = None
        try:
            from unitree_sdk2py.dex3 import Dex3Client
            try:
                self._dex3 = Dex3Client(hand=hand, interface=iface)
            except Exception as exc:
                print(f"[run_geoff_gui] Dex3 connection failed on interface '{iface}':", exc, file=sys.stderr)
                try:
                    self._dex3 = Dex3Client(hand=hand, interface=None)
                    print('[run_geoff_gui] Dex3 connected via auto-detected NIC.')
                except Exception as exc2:
                    print('[run_geoff_gui] Dex3 auto-detect failed:', exc2, file=sys.stderr)
                    self._dex3 = None
            self._hand_poses: dict[str, list[float]] = {}
            try:
                import csv
                csv_path = Path('data/hand_states.csv')
                if csv_path.exists():
                    with csv_path.open('r', newline='') as fp:
                        rdr = csv.DictReader(fp)
                        for row in rdr:
                            label = row.get('label')
                            if label:
                                try:
                                    vals = [float(row[f'joint{i}']) for i in range(7)]
                                    self._hand_poses[label.lower()] = vals
                                except Exception:
                                    pass
            except Exception as exc:
                print('[run_geoff_gui] Could not load hand_states.csv:', exc, file=sys.stderr)
            self._hand_cmd_q: list[float] = [0.0] * 7
            self._hand_pose_seq: list[list[float]] = []
            self._hand_seq_idx: int = 0
            self._HAND_STEP = 0.1
            self._hand_timer = QtCore.QTimer()
            self._hand_timer.setInterval(20)
            self._hand_timer.timeout.connect(self._on_hand_tick)
            self._hand_timer.start()
            self._simple_open_pose = [-0.15717165172100067, -0.41322529315948486, 0.02846403606235981, 0.17782948911190033, -0.025226416066288948, 0.17983606457710266, -0.027690349146723747]
            self._simple_closed_pose = [0.07452802360057831, 0.9478388428688049, 1.766921877861023, -1.4442411661148071, -1.4384468793869019, -1.5298594236373901, -1.4153316020965576]
            self._hand_target: list[float] = list(self._simple_open_pose)
            self._hand_mode: str = 'idle'
            self._GRAB_PRIMARY_IDX: list[int] = [1, 4, 6]
            self._GRAB_TAU: float = getattr(self, '_cli_grip_force', 0.3)
            self._grab_stage: int = 0
            self._hand_open_pose = self._hand_poses.get('open', [0.0] * 7)
            self._hand_closed_pose = self._hand_poses.get('closed', [0.0] * 7)
            self._close_dir = [1.0 if c - o >= 0 else -1.0 for o, c in zip(self._hand_open_pose, self._hand_closed_pose)]
            self._PRESS_TARGET = 0.4
            self._PRESS_HYST = 0.05
            self._log_hand = logging.getLogger('geoff_gui.hand')
            self._PRESS_THR = 0.5
            self._PRESS_MIN_COUNT = 3
        except Exception as exc:
            print('[run_geoff_gui] Dex3 hand control disabled:', exc, file=sys.stderr)
            self._dex3 = None
        self.app.installEventFilter(self)
        self._threads = [threading.Thread(target=_rx_realsense, args=(self._stop_evt,), daemon=True), threading.Thread(target=_run_slam, args=(self._stop_evt,), daemon=True), threading.Thread(target=_rx_battery, args=(self._stop_evt, iface), daemon=True)]
        for t in self._threads:
            t.start()
        self.app.aboutToQuit.connect(self._on_quit)
        try:
            self._configure_arm_variables()
        except Exception as exc:
            print('[run_geoff_gui] Initial arm configuration failed:', exc, file=sys.stderr)

    def _configure_arm_variables(self):
        _WAIST_YAW_IDX = 12
        self._arm_joint_idx = list(range(15, 22)) if self._active_arm == 'left' else list(range(22, 29))
        if not hasattr(self, '_cmd_q'):
            self._cmd_q = {}
        for idx in self._arm_joint_idx:
            self._cmd_q.setdefault(idx, 0.0)
        self._cmd_q.setdefault(_WAIST_YAW_IDX, 0.0)
        for idx in list(self._cmd_q):
            if idx not in self._arm_joint_idx and idx != _WAIST_YAW_IDX:
                self._cmd_q.pop(idx, None)
        if self._active_arm == 'right':
            self._pose_seq = [[(_WAIST_YAW_IDX, 0.0), (22, -0.023), (23, -0.225), (24, +0.502), (25, +1.317), (26, +0.185), (27, +0.125), (28, -0.182)], [(_WAIST_YAW_IDX, 0.0), (22, +0.087), (23, -0.271), (24, +0.323), (25, +0.691), (26, +0.24), (27, -0.771), (28, -0.176)]]
        else:
            self._pose_seq = [[(_WAIST_YAW_IDX, 0.0), (15, +0.211), (16, +0.181), (17, -0.284), (18, +0.672), (19, -0.379), (20, -0.852), (21, -0.019)]]
        self._seq_idx = 0
        self._initialised_from_state = False

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
        self._update_key_overlay()
        with _state_lock:
            rgbd = _state.get('rgbd')
            vx, vy, om = _state.get('vel', (0.0, 0.0, 0.0))
            soc = _state.get('soc')
        if rgbd is not None and rgbd.shape == (480, 1280, 3):
            rgb, depth = (rgbd[:, :640], rgbd[:, 640:])
            px1, px2 = (self._numpy_to_qpix(rgb), self._numpy_to_qpix(depth))
            if px1:
                from PySide6 import QtCore
                scaled = px1.scaled(self.rgb_lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.rgb_lbl.setPixmap(scaled)
            if px2:
                from PySide6 import QtCore
                scaled = px2.scaled(self.depth_lbl.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                self.depth_lbl.setPixmap(scaled)
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
    _SPEED_LIMIT_NORMAL = 0.6
    _SPEED_LIMIT_FAST = 1.2

    def _current_speed_limit(self) -> float:
        return self._SPEED_LIMIT_FAST if self._is_pressed('shift') else self._SPEED_LIMIT_NORMAL

    @staticmethod
    def _clamp(val: float, limit: float) -> float:
        return max(-limit, min(limit, val))
    _DISPLAY_NAMES = {'space': '␣', 'shift': '⇧', 'esc': '⎋', 'up_arrow': '↑', 'down_arrow': '↓', 'left_arrow': '←', 'right_arrow': '→'}

    def _display_name(self, key: str) -> str:
        if key in self._DISPLAY_NAMES:
            return self._DISPLAY_NAMES[key]
        if len(key) == 1:
            return key.upper()
        return key

    def _update_key_overlay(self) -> None:
        if self._pressed:
            if self._fade_anim.state() == QtCore.QAbstractAnimation.Running:
                self._fade_anim.stop()
                self._keys_opacity.setOpacity(1.0)
            keys_txt = '  '.join((self._display_name(k) for k in sorted(self._pressed)))
            self._keys_lbl.setText(keys_txt)
            self._key_overlay.adjustSize()
            self._key_overlay.move(10, 10)
            self._key_overlay.raise_()
        elif self._fade_anim.state() != QtCore.QAbstractAnimation.Running and self._keys_lbl.text() != '–':
            self._fade_anim.stop()
            self._fade_anim.setStartValue(1.0)
            self._fade_anim.setEndValue(0.0)
            self._fade_anim.start()

    def eventFilter(self, _obj, ev):
        from PySide6 import QtCore
        if ev.type() == QtCore.QEvent.KeyPress:
            if ev.isAutoRepeat():
                return False
            key = ev.key()
            name = self._qt_key_name(key, ev.text())
            if name is not None:
                self._pressed.add(name)
                self._maybe_arm_inference(name)
                self._maybe_hand_control(name)
                return True
        elif ev.type() == QtCore.QEvent.KeyRelease:
            if ev.isAutoRepeat():
                return False
            key = ev.key()
            name = self._qt_key_name(key, ev.text())
            if name is None:
                return False
            self._pressed.discard(name)
            return True
        return False

    @staticmethod
    def _qt_key_name(key: int, text: str | None) -> str | None:
        from PySide6 import QtCore
        mapping = {QtCore.Qt.Key_Space: 'space', QtCore.Qt.Key_Escape: 'esc', QtCore.Qt.Key_Z: 'z', QtCore.Qt.Key_Shift: 'shift', QtCore.Qt.Key_Up: 'up_arrow', QtCore.Qt.Key_Down: 'down_arrow', QtCore.Qt.Key_Left: 'left_arrow', QtCore.Qt.Key_Right: 'right_arrow'}
        if key in mapping:
            return mapping[key]
        if text:
            ch = text.lower()
            if ch in ('w', 'a', 's', 'd', 'q', 'e', 'u', 'j', 'f', 'b', 'p', 'o'):
                return ch
            if ch in ('g', 'h'):
                return ch
        return None

    def _is_pressed(self, name: str) -> bool:
        return name in self._pressed

    def _maybe_arm_inference(self, key_name: str) -> None:
        dir_map = {'up_arrow': 'up', 'down_arrow': 'down', 'left_arrow': 'left', 'right_arrow': 'right', 'f': 'forward', 'b': 'back'}
        direction = dir_map.get(key_name)
        if direction is None:
            return
        if self._arm_pub is None:
            return
        try:
            from data.inference_arm import predict_end_positions, load_bundle
            if not hasattr(self, '_arm_bundle_cache'):
                self._arm_bundle_cache = {}
            if self._active_arm not in self._arm_bundle_cache:
                try:
                    from pathlib import Path
                    bundle_path = Path(f'data/artifacts/{self._active_arm}-arm/arm_mlp.joblib')
                    self._arm_bundle_cache[self._active_arm] = load_bundle(bundle_path)
                except TypeError as exc:
                    if '__randomstate_ctor' in str(exc):
                        try:
                            import numpy.random._pickle as _np_pickle

                            def _rs_ctor(*_args, **_kwargs):
                                import numpy as _np
                                return _np.random.RandomState()
                            _np_pickle.__randomstate_ctor = _rs_ctor
                            self._arm_bundle_cache[self._active_arm] = load_bundle(bundle_path)
                        except Exception:
                            raise
                    else:
                        raise
                except Exception:
                    self._arm_bundle_cache[self._active_arm] = load_bundle()
            bundle = self._arm_bundle_cache[self._active_arm]
            start_joints = [self._joint_cur.get(j_idx, self._cmd_q.get(j_idx, 0.0)) for j_idx in sorted(self._arm_joint_idx)]
            preds = predict_end_positions(direction, start_joints, arm=self._active_arm, bundle=bundle)
            target_pose = [(self._waist_idx, self._cmd_q.get(self._waist_idx, 0.0))]
            target_pose += list(zip(sorted(self._arm_joint_idx), preds))
            self._pose_seq = [target_pose]
            self._seq_idx = 0
        except Exception as exc:
            import sys
            print('[run_geoff_gui] Arm inference failed:', exc, file=sys.stderr)

    def _maybe_hand_control(self, key_name: str) -> None:
        if self._dex3 is None:
            return
        if key_name in ('p', 'o'):
            cur_state = self._dex3.read_state(timeout=0.05)
            if cur_state is not None:
                try:
                    cur = [ms.q for ms in cur_state.motor_state[:7]]
                except Exception:
                    cur = list(self._hand_cmd_q)
            else:
                cur = list(self._hand_cmd_q)
            if key_name == 'p':
                target = self._simple_closed_pose
                self._hand_mode = 'closing'
                self._hand_target = list(target)
            else:
                target = self._simple_open_pose
                self._hand_mode = 'opening'
                self._hand_target = list(target)
            self._hand_pose_seq.clear()
            self._hand_seq_idx = 0
            return
        if not getattr(self, '_hand_poses', None):
            return
        if key_name in ('p', 'o'):
            if self._hand_mode in ('closing', 'holding') and key_name == 'p':
                return
            if self._hand_mode == 'opening' and key_name == 'o':
                return
            if key_name == 'p':
                self._log_hand.info('Adaptive grasp triggered')
                target_label = 'closed'
                middle = self._hand_poses.get('middle')
                target = self._hand_poses.get(target_label)
                if middle is None or target is None:
                    return
                cur_state = self._dex3.read_state(timeout=0.05)
                if cur_state is not None:
                    try:
                        cur = [ms.q for ms in cur_state.motor_state[:7]]
                    except Exception:
                        cur = list(self._hand_cmd_q)
                else:
                    cur = list(self._hand_cmd_q)
                self._hand_pose_seq = [cur]
                self._hand_seq_idx = 0
                self._hand_mode = 'adaptive'
            else:
                print('[Dex3] adaptive open', file=sys.stderr)
                target = self._hand_poses.get('open')
                if target is None:
                    return
                cur_state = self._dex3.read_state(timeout=0.05)
                if cur_state is not None:
                    try:
                        cur = [ms.q for ms in cur_state.motor_state[:7]]
                    except Exception:
                        cur = list(self._hand_cmd_q)
                else:
                    cur = list(self._hand_cmd_q)
                self._hand_pose_seq = [cur, target]
                self._hand_seq_idx = 1
                self._hand_mode = 'opening'
            return
        if key_name == 'g':
            if self._hand_mode != 'grabbing':
                self._hand_mode = 'grabbing'
                self._hand_pose_seq = []
                self._hand_seq_idx = 0
                self._log_hand.info('Continuous grab initiated (torque=%.2f N·m)', self._GRAB_TAU)
            return
        if key_name == 'h':
            target_label = 'open'
            middle = self._hand_poses.get('middle')
            target = self._hand_poses.get(target_label)
            if target is None or middle is None:
                print("[run_geoff_gui] Missing hand pose for 'open' or 'middle'.")
                return
            cur_state = self._dex3.read_state(timeout=0.05)
            if cur_state is not None:
                try:
                    cur = [ms.q for ms in cur_state.motor_state[:7]]
                except Exception:
                    cur = list(self._hand_cmd_q)
            else:
                cur = list(self._hand_cmd_q)
            self._hand_pose_seq = [cur, middle, target]
            self._hand_seq_idx = 1
            self._hand_mode = 'opening'
            self._log_hand.info('Hand opening sequence queued.')
            return
        return

    def _on_drive_tick(self):
        lim = self._current_speed_limit()
        if self._is_pressed('w') and (not self._is_pressed('s')):
            self._vx = self._clamp(self._vx + self._LIN_STEP, lim)
        elif self._is_pressed('s') and (not self._is_pressed('w')):
            self._vx = self._clamp(self._vx - self._LIN_STEP, lim)
        else:
            self._vx = 0.0
        if self._is_pressed('q') and (not self._is_pressed('e')):
            self._vy = self._clamp(self._vy + self._LIN_STEP, lim)
        elif self._is_pressed('e') and (not self._is_pressed('q')):
            self._vy = self._clamp(self._vy - self._LIN_STEP, lim)
        else:
            self._vy = 0.0
        if self._is_pressed('a') and (not self._is_pressed('d')):
            self._omega = self._clamp(self._omega + self._ANG_STEP, lim)
        elif self._is_pressed('d') and (not self._is_pressed('a')):
            self._omega = self._clamp(self._omega - self._ANG_STEP, lim)
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

    def _on_arm_tick(self) -> None:
        if self._arm_pub is None:
            return
        if not self._joint_cur and (not getattr(self, '_no_fb_deadline', None)):
            self._no_fb_deadline = time.time() + 1.0
        if not self._joint_cur and time.time() < self._no_fb_deadline:
            return
        if self._seq_idx >= len(self._pose_seq):
            target_pose = self._pose_seq[-1]
        else:
            target_pose = self._pose_seq[self._seq_idx]
        all_reached = True
        if not self._initialised_from_state and self._joint_cur:
            for j_idx, q_val in self._joint_cur.items():
                self._cmd_q[j_idx] = q_val
            self._initialised_from_state = True
        for idx, tgt in target_pose:
            cur = self._cmd_q.get(idx, 0.0)
            diff = tgt - cur
            if abs(diff) <= self._SEQ_EPS:
                self._cmd_q[idx] = tgt
            else:
                step = self._STEP if diff > 0 else -self._STEP
                if abs(step) > abs(diff):
                    step = diff
                self._cmd_q[idx] = cur + step
                all_reached = False
        if all_reached and self._seq_idx < len(self._pose_seq):
            self._seq_idx += 1
        try:
            for idx, q in self._cmd_q.items():
                mc = self._arm_cmd.motor_cmd[idx]
                mc.q = q
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 60.0
                mc.kd = 1.5
            self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)
            self._arm_pub.Write(self._arm_cmd)
        except Exception as exc:
            print('[run_geoff_gui] Arm publish failed:', exc, file=sys.stderr)

    def _on_hand_tick(self):
        if self._dex3 is None:
            return
        if self._hand_mode not in ('closing', 'opening', 'grabbing') and (not self._hand_pose_seq):
            return
        if self._hand_mode in ('closing', 'opening'):
            target = list(self._hand_target)
        if self._hand_mode == 'grabbing':
            closed_pose = self._hand_closed_pose
            target = list(self._hand_cmd_q)
            for j in range(7):
                cur = self._hand_cmd_q[j]
                tgt = closed_pose[j]
                diff = tgt - cur
                if abs(diff) > self._SEQ_EPS:
                    step = self._HAND_STEP if diff > 0 else -self._HAND_STEP
                    if abs(step) > abs(diff):
                        step = diff
                    target[j] = cur + step
                else:
                    target[j] = tgt
        elif self._hand_seq_idx >= len(self._hand_pose_seq):
            target = self._hand_pose_seq[-1] if self._hand_pose_seq else list(self._hand_cmd_q)
        else:
            target = self._hand_pose_seq[self._hand_seq_idx]
        if self._hand_mode == 'adaptive':
            state = self._dex3.read_state(timeout=0.0)
            pressures = []
            if state is not None:
                try:
                    for ps in state.press_sensor_state:
                        pressures.extend(list(ps.pressure))
                except Exception:
                    pass
            self._log_hand.debug('press=%s cmd=%s', pressures[:12], [round(q, 2) for q in self._hand_cmd_q])
            avg_press = sum(pressures) / len(pressures) if pressures else 0.0
            if avg_press < self._PRESS_TARGET:
                target = self._hand_poses.get('closed', self._hand_cmd_q)
            else:
                target = self._hand_cmd_q
            try:
                state = self._dex3.read_state(timeout=0.0)
                if state is not None:
                    pressures = []
                    for ps in state.press_sensor_state:
                        try:
                            pressures.extend(list(ps.pressure))
                        except Exception:
                            pass
                    tip_idx = [2, 5, 8, 11] if len(pressures) >= 12 else list(range(len(pressures)))
                    cnt = sum((1 for i in tip_idx if i < len(pressures) and pressures[i] >= self._PRESS_THR))
                    if cnt >= self._PRESS_MIN_COUNT:
                        self._hand_pose_seq = [list(self._hand_cmd_q)]
                        self._hand_seq_idx = 1
                        self._hand_mode = 'holding'
                        target = list(self._hand_cmd_q)
            except Exception as exc:
                print('[run_geoff_gui] adaptive grasp sensor read failed:', exc, file=sys.stderr)
        all_reached = True
        for i, tgt in enumerate(target):
            cur = self._hand_cmd_q[i]
            diff = tgt - cur
            if abs(diff) <= self._HAND_STEP:
                self._hand_cmd_q[i] = tgt
            else:
                step = self._HAND_STEP if diff > 0 else -self._HAND_STEP
                self._hand_cmd_q[i] = cur + step
                all_reached = False
        if self._hand_mode == 'grabbing':
            all_reached = False
        elif all_reached and self._hand_seq_idx < len(self._hand_pose_seq):
            self._hand_seq_idx += 1
        if self._hand_mode == 'opening' and all_reached and (self._hand_seq_idx >= len(self._hand_pose_seq)):
            self._hand_mode = 'idle'
        try:
            cmd = self._dex3._make_zero_cmd()
            mins, maxs = self._dex3._limits()
            kp = 8.0
            kd = 1.5
            for i, q in enumerate(self._hand_cmd_q):
                mode = self._dex3._pack_mode(i, status=1, timeout=False)
                mc = cmd.motor_cmd[i]
                mc.mode = mode
                mc.kp = kp
                mc.kd = kd
                if self._hand_mode in ('grabbing', 'closing', 'holding'):
                    mc.tau = max(0.3, self._GRAB_TAU) * self._close_dir[i]
                else:
                    mc.tau = 0.0
                mc.q = max(min(q, maxs[i]), mins[i])
            ok = self._dex3._publish(cmd)
            if not ok:
                if not getattr(self, '_dex3_no_match_warned', False):
                    print('[Dex3] WARNING – no subscriber matched for rt/dex3 cmd; retrying…', file=sys.stderr)
                    self._dex3_no_match_warned = True
                for _ in range(3):
                    if self._dex3._publish(cmd):
                        break
        except Exception as exc:
            print('[run_geoff_gui] Dex3 publish failed:', exc, file=sys.stderr)
            self._arm_pub = None

    def _on_damp_pressed(self):
        if self._arm_pub is None:
            print('[run_geoff_gui] Damp request – SDK unavailable')
            return
        try:
            if hasattr(self, '_arm_timer'):
                self._arm_timer.stop()
        except Exception:
            pass
        try:
            for idx in (*range(15, 22), *range(22, 29)):
                mc = self._arm_cmd.motor_cmd[idx]
                mc.q = self._cmd_q.get(idx, 0.0)
                mc.dq = 0.0
                mc.tau = 0.0
                mc.kp = 0.0
                mc.kd = 0.0
            waist_idx = getattr(self, '_waist_idx', 12)
            mc_w = self._arm_cmd.motor_cmd[waist_idx]
            mc_w.q = 0.0
            mc_w.dq = 0.0
            mc_w.tau = 0.0
            mc_w.kp = 60.0
            mc_w.kd = 1.5
            self._cmd_q[waist_idx] = 0.0
            self._arm_cmd.crc = self._crc.Crc(self._arm_cmd)
            self._arm_pub.Write(self._arm_cmd)
            print('[run_geoff_gui] Arms damped, waist centred.')
        except Exception as exc:
            print('[run_geoff_gui] Damp request failed:', exc, file=sys.stderr)

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
        if hasattr(self, '_arm_timer'):
            self._arm_timer.stop()
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
        try:
            from live_slam import _R_MOUNT as _LS_R_MOUNT
            if _LS_R_MOUNT is not None:
                rot = rot @ _LS_R_MOUNT[:3, :3]
        except Exception:
            pass
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
    parser.add_argument('--arm', choices=['left', 'right'], default='left', help='Which arm to control on startup (default: left)')
    parser.add_argument('--hand', choices=['left', 'right'], default='left', help='Which Dex3 hand is connected (default: left).  Pass --hand right when a right-hand unit is attached.')
    parser.add_argument('--grip-force', type=float, dest='grip_force', metavar='N·m', default=0.3, help='Feed-forward torque (approx. N·m) applied during continuous grabs (default: 0.3)')
    args = parser.parse_args()
    window = GeoffWindow(args.iface, args.clear, hand=args.hand, grip_force=args.grip_force)
    window._active_arm = args.arm
    window._arm_selector.setCurrentIndex(0 if args.arm == 'left' else 1)
    try:
        window._configure_arm_variables()
    except Exception as exc:
        print('[run_geoff_gui] Initial arm switch failed:', exc, file=sys.stderr)
    window.run()
if __name__ == '__main__':
    main()
