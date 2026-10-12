from __future__ import annotations
import sys
import time
from typing import Optional
import cv2
try:
    import pyrealsense2 as rs
except ImportError as exc:
    raise SystemExit("pyrealsense2 is not installed. Install it with 'pip install pyrealsense2'") from exc

def colourise_depth(depth_frame: rs.depth_frame) -> cv2.Mat:
    depth_image = cv2.cvtColor(cv2.convertScaleAbs(depth_frame.get_data(), alpha=0.03), cv2.COLOR_GRAY2BGR)
    depth_coloured = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
    return depth_coloured

def get_first_device(context: rs.context) -> Optional[rs.device]:
    devices = context.query_devices()
    if len(devices) == 0:
        return None
    return devices[0]

def run(rgb_width: int=640, rgb_height: int=480, fps: int=30, enable_infra: bool=False, enable_imu: bool=False):
    ctx = rs.context()
    device = get_first_device(ctx)
    if device is None:
        raise RuntimeError('No RealSense device found. Plug in a camera and try again.')
    print('Found device:', device.get_info(rs.camera_info.name))
    print('  Serial number:', device.get_info(rs.camera_info.serial_number))
    print('  Firmware ver.:', device.get_info(rs.camera_info.firmware_version))
    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_stream(rs.stream.depth, rgb_width, rgb_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, rgb_width, rgb_height, rs.format.bgr8, fps)
    if enable_infra:
        config.enable_stream(rs.stream.infrared, 1, rgb_width, rgb_height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, rgb_width, rgb_height, rs.format.y8, fps)
    if enable_imu:
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    spatial_filter = rs.spatial_filter()
    temporal_filter = rs.temporal_filter()
    align_to = rs.stream.color
    align = rs.align(align_to)
    print('Starting pipeline …')
    profile = pipeline.start(config)
    print('Camera intrinsics (colour stream):')
    colour_intr: rs.video_stream_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = colour_intr.get_intrinsics()
    print(f'  Resolution    : {intr.width} × {intr.height}')
    print(f'  Focal length  : fx={intr.fx:.1f}  fy={intr.fy:.1f}')
    print(f'  Principal pt. : cx={intr.ppx:.1f} cy={intr.ppy:.1f}')
    last_time = time.perf_counter()
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame: rs.depth_frame = aligned_frames.get_depth_frame()
            colour_frame: rs.video_frame = aligned_frames.get_color_frame()
            if not depth_frame or not colour_frame:
                continue
            depth_frame = spatial_filter.process(depth_frame)
            depth_frame = temporal_filter.process(depth_frame)
            colour_image = colour_frame.get_data()
            depth_coloured = colourise_depth(depth_frame)
            combined = cv2.hconcat([colour_image, depth_coloured])
            now = time.perf_counter()
            fps_calc = 1.0 / (now - last_time)
            last_time = now
            cv2.putText(combined, f'FPS: {fps_calc:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.imshow('RealSense RGB + Depth', combined)
            if enable_infra:
                ir_left = aligned_frames.get_infrared_frame(1)
                ir_right = aligned_frames.get_infrared_frame(2)
                if ir_left and ir_right:
                    ir_left_img = ir_left.get_data()
                    ir_right_img = ir_right.get_data()
                    cv2.imshow('IR-left', ir_left_img)
                    cv2.imshow('IR-right', ir_right_img)
            if enable_imu:
                gyro: rs.motion_frame = frames.first_or_default(rs.stream.gyro)
                accel: rs.motion_frame = frames.first_or_default(rs.stream.accel)
                if gyro and accel:
                    g_data = gyro.as_motion_frame().get_motion_data()
                    a_data = accel.as_motion_frame().get_motion_data()
                    print(f'Gyro [rad/s]: x={g_data.x:+.3f} y={g_data.y:+.3f} z={g_data.z:+.3f} | Accel [m/s²]: x={a_data.x:+.3f} y={a_data.y:+.3f} z={a_data.z:+.3f}', end='\r')
            key = cv2.waitKey(1) & 255
            if key in (ord('q'), 27):
                break
    finally:
        print('\nStopping pipeline, closing windows …')
        pipeline.stop()
        cv2.destroyAllWindows()
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Simple RealSense viewer (colour + depth) written in Python', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--width', type=int, default=640, help='Width of the RGB/depth stream')
    parser.add_argument('--height', type=int, default=480, help='Height of the RGB/depth stream')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate')
    parser.add_argument('--infra', action='store_true', help='Also display the two IR streams')
    parser.add_argument('--imu', action='store_true', help='Print IMU (gyro + accel) readings')
    args = parser.parse_args()
    try:
        run(rgb_width=args.width, rgb_height=args.height, fps=args.fps, enable_infra=args.infra, enable_imu=args.imu)
    except RuntimeError as err:
        sys.exit(str(err))
