from __future__ import annotations
import ctypes as _C
import json
import os
import sys
import threading
import time
from ctypes import POINTER, c_char_p, c_uint8, c_uint16, c_uint32, c_float, c_bool
from pathlib import Path
from typing import Optional
import numpy as np

def _load_lib():
    for name in ('liblivox_lidar_sdk_shared.so', 'liblivox_lidar_sdk.so', 'livox_lidar_sdk.dll'):
        try:
            return _C.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise OSError('liblivox_lidar_sdk shared library not found.  Build & install Livox-SDK2 first (see wrapper docstring).')
_lib = _load_lib()

class _LivoxLidarEthernetPacket(_C.Structure):
    _pack_ = 1
    _fields_ = [('version', c_uint8), ('length', c_uint16), ('time_interval', c_uint16), ('dot_num', c_uint16), ('udp_cnt', c_uint16), ('frame_cnt', c_uint8), ('data_type', c_uint8), ('time_type', c_uint8), ('rsvd', c_uint8 * 12), ('crc32', c_uint32), ('timestamp', c_uint8 * 8), ('data', c_uint8 * 1)]

class _CartesianHighPoint(_C.Structure):
    _pack_ = 1
    _fields_ = [('x', _C.c_int32), ('y', _C.c_int32), ('z', _C.c_int32), ('reflectivity', c_uint8), ('tag', c_uint8)]
_PointCb = _C.CFUNCTYPE(None, c_uint32, c_uint8, POINTER(_LivoxLidarEthernetPacket), _C.c_void_p)

class _LivoxLidarInfo(_C.Structure):
    _fields_ = [('dev_type', c_uint8), ('sn', _C.c_char * 16), ('lidar_ip', _C.c_char * 16)]
_InfoChangeCb = _C.CFUNCTYPE(None, c_uint32, POINTER(_LivoxLidarInfo), _C.c_void_p)
_lib.SetLivoxLidarInfoChangeCallback.argtypes = (_InfoChangeCb, _C.c_void_p)
_lib.SetLivoxLidarWorkMode.argtypes = (c_uint32, c_uint8, _C.c_void_p, _C.c_void_p)
_lib.SetLivoxLidarWorkMode.restype = c_uint32
_lib.EnableLivoxLidarPointSend.argtypes = (c_uint32, _C.c_void_p, _C.c_void_p)
_lib.EnableLivoxLidarPointSend.restype = c_uint32
_lib.SetLivoxLidarPclDataType.argtypes = (c_uint32, c_uint8, _C.c_void_p, _C.c_void_p)
_lib.LivoxLidarAddPointCloudObserver.argtypes = (_PointCb, _C.c_void_p)
_lib.LivoxLidarAddPointCloudObserver.restype = c_uint16
_lib.LivoxLidarSdkInit.argtypes = (c_char_p, c_char_p, _C.c_void_p)
_lib.LivoxLidarSdkInit.restype = c_bool
_lib.LivoxLidarSdkStart.argtypes = ()
_lib.LivoxLidarSdkStart.restype = c_bool
_lib.LivoxLidarSdkUninit.argtypes = ()
_lib.LivoxLidarSdkUninit.restype = None
_lib.SetLivoxLidarPointCloudCallBack.argtypes = (_PointCb, _C.c_void_p)

class Livox2:

    def __init__(self, config_path: str | Path, host_ip: str, *, frame_time: float=0.2, frame_packets: int=120):
        self._config_path = os.fspath(config_path).encode()
        if not _lib.LivoxLidarSdkInit(self._config_path, host_ip.encode(), None):
            raise RuntimeError('LivoxLidarSdkInit failed – check config path & JSON')
        self._cb = _PointCb(self._on_packet)
        _lib.SetLivoxLidarPointCloudCallBack(self._cb, None)
        _lib.LivoxLidarSdkStart()
        self._info_cb = _InfoChangeCb(self._on_info_change)
        _lib.SetLivoxLidarInfoChangeCallback(self._info_cb, None)
        self._running = True
        self._frame_time = float(frame_time)
        self._frame_packets = int(frame_packets)

    def spin(self):
        try:
            while self._running:
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        if self._running:
            _lib.LivoxLidarSdkUninit()
            self._running = False

    def handle_points(self, xyz: np.ndarray):
        print(f'frame {len(xyz)} pts')

    def _on_packet(self, handle: int, dev_type: int, pkt_ptr, _client):
        pkt = pkt_ptr.contents
        n = pkt.dot_num
        if n == 0:
            return
        if pkt.data_type == 1:
            _Arr = _CartesianHighPoint * n
            points = _C.cast(pkt.data, POINTER(_Arr)).contents
            arr = np.ctypeslib.as_array(points)
            xyz = np.stack((arr['x'], arr['y'], arr['z']), axis=1).astype(np.float32) / 1000.0
        elif pkt.data_type == 2:

            class _LowPoint(_C.Structure):
                _fields_ = [('x', _C.c_int16), ('y', _C.c_int16), ('z', _C.c_int16), ('reflectivity', c_uint8), ('tag', c_uint8)]
            _ArrL = _LowPoint * n
            pts = _C.cast(pkt.data, POINTER(_ArrL)).contents
            arr = np.ctypeslib.as_array(pts)
            xyz = np.stack((arr['x'], arr['y'], arr['z']), axis=1).astype(np.float32) / 100.0
        else:
            return
        state = self.__dict__.setdefault('_frame_state', {})
        buf, last_t = state.get(handle, ([], time.time()))
        buf.append(xyz)
        now = time.time()
        elapsed = now - last_t
        if elapsed >= self._frame_time or len(buf) >= self._frame_packets:
            frame_xyz = np.concatenate(buf, axis=0)
            try:
                self.handle_points(frame_xyz)
            except Exception as exc:
                print('Exception in handle_points:', exc, file=sys.stderr)
            print(f'[Livox2] frame {frame_xyz.shape[0]} pts  (Δt={elapsed * 1000:.1f} ms)')
            buf = []
            last_t = now
        state[handle] = (buf, last_t)

    def _on_info_change(self, handle: int, info_ptr, _client):
        print(f'[Livox2] InfoChange handle={handle}')
        kNormal = 1
        _lib.SetLivoxLidarWorkMode(handle, kNormal, None, None)
        _lib.EnableLivoxLidarPointSend(handle, None, None)
        _lib.SetLivoxLidarPclDataType(handle, 1, None, None)
if __name__ == '__main__':
    cfg = Path('mid360_config.json')
    if not cfg.exists():
        host_ip = os.environ.get('HOST_IP', '192.168.123.222')
        data = {'MID360': {'lidar_net_info': {'cmd_data_port': 56100, 'push_msg_port': 56200, 'point_data_port': 56300, 'imu_data_port': 56400, 'log_data_port': 56500}, 'host_net_info': [{'host_ip': host_ip, 'multicast_ip': '224.1.1.5', 'cmd_data_port': 56101, 'push_msg_port': 56201, 'point_data_port': 56301, 'imu_data_port': 56401, 'log_data_port': 56501}]}}
        cfg.write_text(json.dumps(data, indent=2))
        print('[Livox2] Wrote default mid360_config.json with host_ip', host_ip)
    lidar = Livox2(cfg, host_ip='192.168.123.222')
    lidar.spin()
