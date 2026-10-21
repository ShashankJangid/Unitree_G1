"""Single-threaded GLFW renderer for Gangnam playback."""

import os
import sys
import threading
import time

import mujoco
import numpy as np


def run_viewer(runtime, count) -> None:
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Gangnam viewer must run on the main thread")
    if sys.platform.startswith("linux") and os.environ.get("DISPLAY") and "PYGLFW_LIBRARY_VARIANT" not in os.environ:
        os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"
    import glfw

    window = None
    context = None
    try:
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        window = glfw.create_window(1280, 720, "Gangnam Style — G1", None, None)
        if not window:
            raise RuntimeError("GLFW window creation failed")
        glfw.make_context_current(window)
        scene = mujoco.MjvScene(runtime.model, maxgeom=10000)
        context = mujoco.MjrContext(runtime.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        camera = mujoco.MjvCamera()
        camera.distance = 3.7
        camera.elevation = -18
        camera.azimuth = 135
        follow_position = np.asarray(runtime.data.xpos[runtime.pelvis]).copy()
        camera.lookat[:] = follow_position
        option = mujoco.MjvOption()
        viewport = mujoco.MjrRect(0, 0, 0, 0)
        buttons = {"left": False, "right": False}
        cursor = [0., 0.]

        def move_camera(action, dx, dy):
            # MuJoCo 3.16 removed the scene argument; older supported versions
            # still require it. Keep both public signatures usable.
            try:
                mujoco.mjv_moveCamera(runtime.model, action, dx, dy, camera)
            except TypeError:
                mujoco.mjv_moveCamera(runtime.model, action, dx, dy, scene, camera)

        def key_callback(win, key, scancode, action, mods):
            if action == glfw.PRESS and key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
                glfw.set_window_should_close(win, True)

        def button_callback(win, button, action, mods):
            if button == glfw.MOUSE_BUTTON_LEFT:
                buttons["left"] = action == glfw.PRESS
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                buttons["right"] = action == glfw.PRESS
            cursor[:] = glfw.get_cursor_pos(win)

        def cursor_callback(win, x, y):
            dx, dy = x - cursor[0], y - cursor[1]
            cursor[:] = x, y
            if glfw.window_should_close(win):
                return
            height = max(1, glfw.get_window_size(win)[1])
            if buttons["left"]:
                move_camera(mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx / height, 0)
                move_camera(mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0, dy / height)
            elif buttons["right"]:
                move_camera(mujoco.mjtMouse.mjMOUSE_MOVE_H, dx / height, 0)
                move_camera(mujoco.mjtMouse.mjMOUSE_MOVE_V, 0, dy / height)

        def scroll_callback(win, dx, dy):
            if not glfw.window_should_close(win):
                camera.distance = max(.3, min(30., camera.distance * (0.9 ** dy)))

        glfw.set_key_callback(window, key_callback)
        glfw.set_mouse_button_callback(window, button_callback)
        glfw.set_cursor_pos_callback(window, cursor_callback)
        glfw.set_scroll_callback(window, scroll_callback)

        dt = runtime.model.opt.timestep
        start = time.monotonic()
        next_frame = start
        while (count is None or runtime.steps < count) and not glfw.window_should_close(window):
            glfw.poll_events()
            if glfw.window_should_close(window):
                break
            now = time.monotonic()
            # Catch up to wall time without skipping any physics step; bound each batch
            # so event polling and rendering remain responsive after a stall.
            due = min(8, 8 if count is None else count - runtime.steps,
                      max(0, int((now - start) / dt) - runtime.steps))
            for _ in range(due):
                runtime.step_physics()
            if now >= next_frame and not glfw.window_should_close(window):
                position = np.asarray(runtime.data.xpos[runtime.pelvis]).copy()
                camera.lookat[:] = np.asarray(camera.lookat) + position - follow_position
                follow_position = position
                width, height = glfw.get_framebuffer_size(window)
                if width > 0 and height > 0:
                    viewport.width, viewport.height = width, height
                    mujoco.mjv_updateScene(runtime.model, runtime.data, option, None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
                    mujoco.mjr_render(viewport, scene, context)
                    if not glfw.window_should_close(window):
                        glfw.swap_buffers(window)
                next_frame = now + 1 / 60
            else:
                time.sleep(min(.002, max(0., min(start + (runtime.steps + 1) * dt, next_frame) - time.monotonic())))
    finally:
        try:
            if window:
                glfw.make_context_current(window)
                try:
                    if context is not None:
                        context.free()
                finally:
                    glfw.destroy_window(window)
        finally:
            glfw.terminate()
