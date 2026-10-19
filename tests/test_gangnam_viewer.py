import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gangnam_viewer


def setup_viewer(monkeypatch, *, close=False, fail_window=False, fail_step=False):
    events = []
    state = {"closed": close, "destroyed": False}
    callbacks = {}

    def record(name):
        assert not state["destroyed"] or name == "terminate"
        events.append((name, threading.get_ident()))

    def init():
        record("init")
        return True

    def create_window(*args):
        record("create")
        return None if fail_window else object()

    def destroy_window(win):
        record("destroy")
        state["destroyed"] = True

    def terminate():
        record("terminate")

    def poll_events():
        record("poll")
        if close:
            state["closed"] = True

    glfw = SimpleNamespace(init=init, create_window=create_window,
        make_context_current=lambda win: record("current"), destroy_window=destroy_window,
        terminate=terminate, poll_events=poll_events,
        window_should_close=lambda win: state["closed"],
        swap_buffers=lambda win: record("swap"),
        get_framebuffer_size=lambda win: (800, 600),
        set_key_callback=lambda win, cb: callbacks.update(key=cb),
        set_mouse_button_callback=lambda win, cb: callbacks.update(button=cb),
        set_cursor_pos_callback=lambda win, cb: callbacks.update(cursor=cb),
        set_scroll_callback=lambda win, cb: callbacks.update(scroll=cb),
        get_cursor_pos=lambda win: (0, 0), get_window_size=lambda win: (800, 600),
        MOUSE_BUTTON_LEFT=0, MOUSE_BUTTON_RIGHT=1, PRESS=1, KEY_ESCAPE=256, KEY_Q=81,
        set_window_should_close=lambda win, value: state.update(closed=value))
    monkeypatch.setitem(sys.modules, "glfw", glfw)

    class Context:
        def __init__(self, *args):
            record("context")

        def free(self):
            record("free")

    class Camera:
        def __init__(self):
            self.lookat = [0, 0, 0]

    graphics = SimpleNamespace(MjvScene=lambda *args, **kwargs: object(),
        MjrContext=Context, MjvCamera=Camera, MjvOption=lambda: object(),
        MjrRect=lambda *args: SimpleNamespace(width=0, height=0),
        mjtFontScale=SimpleNamespace(mjFONTSCALE_150=150),
        mjtCatBit=SimpleNamespace(mjCAT_ALL=1),
        mjtMouse=SimpleNamespace(mjMOUSE_ROTATE_H=2, mjMOUSE_ROTATE_V=1,
                                mjMOUSE_MOVE_H=4, mjMOUSE_MOVE_V=3),
        mjv_moveCamera=lambda model, action, dx, dy, camera: record("move"),
        mjv_updateScene=lambda *args: record("scene"),
        mjr_render=lambda *args: record("render"))
    monkeypatch.setattr(gangnam_viewer, "mujoco", graphics)
    data = SimpleNamespace(xpos=[[0, 0, 0]])
    runtime = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)),
        data=data, pelvis=0, steps=0)

    def step():
        record("step")
        if fail_step:
            raise ValueError("physics failed")
        runtime.steps += 1

    runtime.step_physics = step
    runtime.fake_callbacks = callbacks
    runtime.fake_graphics = graphics
    return runtime, events


def test_complete_playback_cleanup_order_and_thread(monkeypatch):
    runtime, events = setup_viewer(monkeypatch)
    monkeypatch.setattr(gangnam_viewer.time, "monotonic", lambda: 100.)
    # A pre-existing step count makes the absolute count terminate immediately.
    runtime.steps = 3
    gangnam_viewer.run_viewer(runtime, 3)
    names = [name for name, _ in events]
    assert names.index("free") < names.index("destroy") < names.index("terminate")
    assert {tid for _, tid in events} == {threading.get_ident()}


def test_early_close_never_swaps(monkeypatch):
    runtime, events = setup_viewer(monkeypatch, close=True)
    gangnam_viewer.run_viewer(runtime, 5)
    assert runtime.steps == 0
    assert "swap" not in [name for name, _ in events]


def test_physics_exception_cleans_up(monkeypatch):
    runtime, events = setup_viewer(monkeypatch, fail_step=True)
    calls = iter([100., 100.02])
    monkeypatch.setattr(gangnam_viewer.time, "monotonic", lambda: next(calls, 100.02))
    with pytest.raises(ValueError, match="physics failed"):
        gangnam_viewer.run_viewer(runtime, 5)
    names = [name for name, _ in events]
    assert names.index("free") < names.index("destroy") < names.index("terminate")


def test_window_failure_terminates(monkeypatch):
    runtime, events = setup_viewer(monkeypatch, fail_window=True)
    with pytest.raises(RuntimeError, match="window creation"):
        gangnam_viewer.run_viewer(runtime, 5)
    assert [name for name, _ in events] == ["init", "create", "terminate"]


def test_main_thread_required(monkeypatch):
    runtime, events = setup_viewer(monkeypatch)
    monkeypatch.setattr(gangnam_viewer.threading, "main_thread", lambda: object())
    with pytest.raises(RuntimeError, match="main thread"):
        gangnam_viewer.run_viewer(runtime, 1)
    assert events == []


def test_playback_and_mouse_controls_use_current_camera_api(monkeypatch):
    runtime, events = setup_viewer(monkeypatch)
    tick = iter([100., 100.02])
    monkeypatch.setattr(gangnam_viewer.time, "monotonic", lambda: next(tick, 100.02))

    def poll():
        cb = runtime.fake_callbacks
        cb["button"](None, 1, 1, 0)
        cb["cursor"](None, 20, 10)
        cb["button"](None, 0, 1, 0)
        cb["cursor"](None, 30, 15)
        cb["scroll"](None, 0, 2)

    monkeypatch.setattr(sys.modules["glfw"], "poll_events", poll)
    gangnam_viewer.run_viewer(runtime, 5)
    names = [name for name, _ in events]
    assert runtime.steps == 5
    assert names.count("move") == 4
    assert names.index("render") < names.index("swap") < names.index("free")


def test_close_event_skips_render_and_swap(monkeypatch):
    runtime, events = setup_viewer(monkeypatch)
    monkeypatch.setattr(sys.modules["glfw"], "poll_events",
                        lambda: runtime.fake_callbacks["key"](None, 256, 0, 1, 0))
    gangnam_viewer.run_viewer(runtime, 5)
    assert runtime.steps == 0
    assert not {"render", "swap", "step"}.intersection(name for name, _ in events)


def test_keyboard_interrupt_frees_context_before_window(monkeypatch):
    runtime, events = setup_viewer(monkeypatch)

    def interrupt():
        raise KeyboardInterrupt

    monkeypatch.setattr(sys.modules["glfw"], "poll_events", interrupt)
    with pytest.raises(KeyboardInterrupt):
        gangnam_viewer.run_viewer(runtime, 5)
    names = [name for name, _ in events]
    assert names.index("free") < names.index("destroy") < names.index("terminate")


def test_unbounded_viewer_closes_and_cleans_up_once(monkeypatch):
    runtime, events = setup_viewer(monkeypatch)
    ticks = iter([100., 100.02, 100.04])
    monkeypatch.setattr(gangnam_viewer.time, "monotonic", lambda: next(ticks, 100.04))
    polls = 0

    def poll():
        nonlocal polls
        polls += 1
        if polls == 2:
            runtime.fake_callbacks["key"](None, 81, 0, 1, 0)

    monkeypatch.setattr(sys.modules["glfw"], "poll_events", poll)
    gangnam_viewer.run_viewer(runtime, None)
    names = [name for name, _ in events]
    assert runtime.steps == 8
    assert names.count("free") == names.count("destroy") == names.count("terminate") == 1
