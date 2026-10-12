from __future__ import annotations
import argparse
import sys
import numpy as np
import cv2
import pyrealsense2 as rs
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GstApp

def gst_pipeline(client_ip: str, w: int, h: int, fps: int) -> tuple[Gst.Pipeline, GstApp.AppSrc, GstApp.AppSrc]:
    Gst.init(None)
    rgb_caps = f'video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1'
    depth_caps = f'video/x-raw,format=BGR,width={w},height={h},framerate={fps}/1'
    launch_description = f'appsrc name=src_rgb is-live=true do-timestamp=true format=time caps={rgb_caps} ! videoconvert ! nvvidconv ! nvv4l2h264enc bitrate=4000000 insert-sps-pps=true idrinterval=15 ! rtph264pay config-interval=1 pt=96 ! udpsink host={client_ip} port=5600 sync=false appsrc name=src_depth is-live=true do-timestamp=true format=time caps={depth_caps} ! videoconvert ! nvvidconv ! nvv4l2h264enc bitrate=2000000 insert-sps-pps=true idrinterval=15 ! rtph264pay config-interval=1 pt=97 ! udpsink host={client_ip} port=5602 sync=false'
    pipeline = Gst.parse_launch(launch_description)
    src_rgb = pipeline.get_by_name('src_rgb')
    src_depth = pipeline.get_by_name('src_depth')
    return (pipeline, src_rgb, src_depth)

def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--client-ip', required=True)
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()
    pipeline, src_rgb, src_depth = gst_pipeline(args.client_ip, args.width, args.height, args.fps)
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    cfg.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    pipe = rs.pipeline()
    pipe.start(cfg)
    temp_filter = rs.temporal_filter()
    pipeline.set_state(Gst.State.PLAYING)
    duration = Gst.util_uint64_scale_int(1, Gst.SECOND, args.fps)
    try:
        while True:
            frames = pipe.wait_for_frames()
            colour = np.asarray(frames.get_color_frame().get_data())
            depth = frames.get_depth_frame()
            depth = temp_filter.process(depth)
            depth16 = np.asarray(depth.get_data())
            depth_clip = np.clip(depth16, 0, 6000)
            depth8 = cv2.convertScaleAbs(depth_clip, alpha=255.0 / 6000.0)
            depth_bgr = cv2.applyColorMap(depth8, cv2.COLORMAP_PLASMA)
            buf_rgb = Gst.Buffer.new_allocate(None, colour.nbytes, None)
            buf_rgb.fill(0, colour.tobytes())
            buf_rgb.duration = duration
            src_rgb.emit('push-buffer', buf_rgb)
            buf_d = Gst.Buffer.new_allocate(None, depth_bgr.nbytes, None)
            buf_d.fill(0, depth_bgr.tobytes())
            buf_d.duration = duration
            src_depth.emit('push-buffer', buf_d)
    except KeyboardInterrupt:
        print('Interrupted – shutting down …')
    finally:
        for s in (src_rgb, src_depth):
            s.emit('end-of-stream')
        pipeline.set_state(Gst.State.NULL)
        pipe.stop()
if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('Error:', exc)
        sys.exit(1)
