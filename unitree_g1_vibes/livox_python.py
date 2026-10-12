from __future__ import annotations
import ctypes as _C
import sys
import threading
import time
from ctypes import c_uint8, c_uint32, c_float, c_bool, POINTER, cdll
from typing import Dict
import numpy as np
_CANDIDATES = ('liblivox_sdk.so', 'LivoxSdk.dll')

def _load_library():
    for name in _CANDIDATES:
        try:
            return cdll.LoadLibrary(name)
        except OSError:
            continue
    raise OSError('Could not locate Livox shared library (liblivox_sdk.so / LivoxSdk.dll). Build the Livox-SDK first, or add its folder to LD_LIBRARY_PATH / PATH.')
_lib = _load_library()
_kBroadcastCodeSize = 16

class _LivoxEthPacket(_C.Structure):
    _fields_ = [('version', c_uint8), ('slot', c_uint8), ('lidar_id', c_uint8), ('rsvd', c_uint8), ('err_code', c_uint32), ('timestamp_type', c_uint8), ('data_type', c_uint8), ('timestamp', c_uint8 * 8), ('data', c_uint8 * 1)]

class _LivoxRawPoint(_C.Structure):
    _fields_ = [('x', _C.c_int32), ('y', _C.c_int32), ('z', _C.c_int32), ('reflectivity', c_uint8)]
_DataCallback = _C.CFUNCTYPE(None, c_uint8, POINTER(_LivoxEthPacket), c_uint32, _C.c_void_p)

class _BroadcastDeviceInfo(_C.Structure):
    _fields_ = [('broadcast_code', _C.c_char * _kBroadcastCodeSize), ('dev_type', c_uint8), ('reserved', _C.c_uint16), ('ip', _C.c_char * 16)]
_BroadcastCallback = _C.CFUNCTYPE(None, POINTER(_BroadcastDeviceInfo))

class _DeviceInfo(_C.Structure):
    _fields_ = [('broadcast_code', _C.c_char * _kBroadcastCodeSize), ('handle', c_uint8), ('slot', c_uint8), ('id', c_uint8), ('type', c_uint8), ('data_port', _C.c_uint16), ('cmd_port', _C.c_uint16), ('sensor_port', _C.c_uint16), ('ip', _C.c_char * 16), ('state', c_uint8), ('feature', c_uint8)]
_DeviceEvent = c_uint8
_DeviceStateCallback = _C.CFUNCTYPE(None, POINTER(_DeviceInfo), _DeviceEvent)
_lib.Init.restype = c_bool
_lib.Start.restype = c_bool
_lib.Uninit.restype = None
_lib.SetBroadcastCallback.argtypes = (_BroadcastCallback,)
_lib.SetDeviceStateUpdateCallback.argtypes = (_DeviceStateCallback,)
_lib.AddLidarToConnect.argtypes = (_C.c_char_p, POINTER(c_uint8))
_lib.AddLidarToConnect.restype = c_uint32
_lib.SetDataCallback.argtypes = (c_uint8, _DataCallback, _C.c_void_p)
_lib.LidarStartSampling.argtypes = (c_uint8, _C.c_void_p, _C.c_void_p)

class Livox:

    def __init__(self):
        if not _lib.Init():
            raise RuntimeError('Livox SDK Init() failed')
        self._cb_broadcast = _BroadcastCallback(self._on_broadcast)
        self._cb_dev_state = _DeviceStateCallback(self._on_device_state)
        self._cb_data = _DataCallback(self._on_data)
        _lib.SetBroadcastCallback(self._cb_broadcast)
        _lib.SetDeviceStateUpdateCallback(self._cb_dev_state)
        if not _lib.Start():
            _lib.Uninit()
            raise RuntimeError('Livox SDK Start() failed')
        self._handles: Dict[int, str] = {}
        self._running = True

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
            self._running = False
            _lib.Uninit()

    def handle_points(self, xyz: np.ndarray):
        print(f'frame {xyz.shape[0]} pts')

    def _on_broadcast(self, info_ptr):
        info = info_ptr.contents
        code = bytes(info.broadcast_code).decode('ascii', 'ignore').rstrip('\x00')
        handle = c_uint8()
        stat = _lib.AddLidarToConnect(code.encode('ascii'), _C.byref(handle))
        if stat != 0:
            print(f'[Livox] AddLidarToConnect failed for {code} (status {stat})')
            return
        _lib.SetDataCallback(handle.value, self._cb_data, None)
        _lib.LidarStartSampling(handle.value, None, None)
        self._handles[handle.value] = code

    def _on_device_state(self, info_ptr, event):
        if not info_ptr:
            return
        info = info_ptr.contents
        if event == 0:
            print(f'[Livox] Device {info.handle} connected. State={info.state}')
            _lib.LidarStartSampling(info.handle, None, None)
        elif event == 1:
            print(f'[Livox] Device {info.handle} disconnected')

    def _on_data(self, handle: int, pkt_ptr, n_points: int, _client):
        if n_points == 0:
            return
        pkt = pkt_ptr.contents
        if pkt.data_type == 0:
            _RawArray = _LivoxRawPoint * n_points
            raw_points = _C.cast(pkt.data, POINTER(_RawArray)).contents
            raw_np = np.ctypeslib.as_array(raw_points)
            xyz_m = np.stack((raw_np['x'], raw_np['y'], raw_np['z']), axis=1).astype(np.float32) / 1000.0
        elif pkt.data_type == 2:

            class _ExtPoint(_C.Structure):
                _fields_ = [('x', _C.c_int32), ('y', _C.c_int32), ('z', _C.c_int32), ('reflectivity', c_uint8), ('tag', c_uint8)]
            _ExtArray = _ExtPoint * n_points
            points = _C.cast(pkt.data, POINTER(_ExtArray)).contents
            p_np = np.ctypeslib.as_array(points)
            xyz_m = np.stack((p_np['x'], p_np['y'], p_np['z']), axis=1).astype(np.float32) / 1000.0
        else:
            return
        try:
            self.handle_points(xyz_m)
        except Exception as exc:
            print('[Livox] Exception inside handle_points:', exc, file=sys.stderr)
if __name__ == '__main__':
    lidar = Livox()

    def _print(x):
        print('Received', x.shape)
    lidar.handle_points = _print
    lidar.spin()
