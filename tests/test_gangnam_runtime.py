from pathlib import Path
import sys
import types

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from g1_gangnam import GangnamRuntime, run_gangnam
from grit_controller import GRIT_JOINT_NAMES


@pytest.fixture
def viewer_module(monkeypatch):
    module = types.ModuleType("gangnam_viewer")
    monkeypatch.setitem(sys.modules, "gangnam_viewer", module)
    return module


def test_named_motor_mapping_and_no_support():
    runtime = GangnamRuntime()
    model, data = runtime.model, runtime.data
    assert model.neq == 1
    assert not np.any(data.eq_active)
    assert [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j)) for j in runtime.jids] == GRIT_JOINT_NAMES
    assert set(runtime.motors) == set(range(29))
    assert np.allclose(data.qpos[runtime.qids], runtime.policy.sample(runtime.policy.start_time)[2])
    assert np.any(data.qvel[:3])
    runtime.step_physics()
    assert not np.any(data.eq_active)
    assert np.all(data.ctrl[runtime.motors] >= runtime.torque_limits[:, 0])
    assert np.all(data.ctrl[runtime.motors] <= runtime.torque_limits[:, 1])


def test_policy_cadence_and_bounded_run():
    runtime = GangnamRuntime()
    calls = []
    infer = runtime.policy.infer

    def counted(*args):
        calls.append(args[0])
        return infer(*args)

    runtime.policy.infer = counted
    for _ in range(30):
        runtime.step_physics()
    assert runtime.steps == 30
    assert np.allclose(calls, [0., .02, .04])
    assert runtime.summary()["elapsed"] == pytest.approx(.06)
    assert not np.any(runtime.data.eq_active)
    assert run_gangnam(headless=True, duration=.04)["steps"] == 20


def test_fall_is_not_reset_or_hidden():
    runtime = GangnamRuntime()
    runtime.data.qpos[2] = .3
    mujoco.mj_forward(runtime.model, runtime.data)
    with pytest.raises(RuntimeError, match="fell"):
        runtime._check_state()
    assert runtime.data.qpos[2] == pytest.approx(.3)


@pytest.mark.parametrize("steps", [4, 20])
def test_graphical_run_returns_actual_summary(monkeypatch, viewer_module, steps):
    calls = []

    def fake_viewer(runtime, count):
        calls.append((runtime, count))
        for _ in range(steps):
            runtime.step_physics()

    monkeypatch.setattr(viewer_module, "run_viewer", fake_viewer, raising=False)
    result = run_gangnam(duration=.04)
    assert len(calls) == 1
    runtime, count = calls[0]
    assert isinstance(runtime, GangnamRuntime)
    assert count == 20
    assert result == runtime.summary()
    assert result["steps"] == steps


def test_graphical_keyboard_interrupt_returns_actual_summary(monkeypatch, viewer_module, capsys):
    def interrupted(runtime, count):
        assert count == 20
        runtime.step_physics()
        raise KeyboardInterrupt

    monkeypatch.setattr(viewer_module, "run_viewer", interrupted, raising=False)
    result = run_gangnam(duration=.04)
    assert result["steps"] == 1
    assert result["elapsed"] == pytest.approx(.002)
    assert "Interrupted by user" in capsys.readouterr().out


def test_graphical_error_propagates(monkeypatch, viewer_module):
    def broken(runtime, count):
        raise RuntimeError("graphics failed")

    monkeypatch.setattr(viewer_module, "run_viewer", broken, raising=False)
    with pytest.raises(RuntimeError, match="graphics failed"):
        run_gangnam(duration=.04)


def test_complete_gangnam_excerpt_without_support_or_resets():
    result = run_gangnam(headless=True)
    assert result["elapsed"] == pytest.approx(20.3)
    assert result["steps"] == 10150
    assert result["min_z"] > .6
    assert result["min_up"] > .9
    assert result["tracking_rms"] < .2
    assert result["completed_cycles"] == 1


def test_multicycle_replay_preserves_global_metrics_and_restarts_policy(capsys):
    runtime = GangnamRuntime()
    calls = []
    infer = runtime.policy.infer

    def observed(*args):
        if args[0] == 0:
            calls.append((runtime.steps, runtime.policy.previous_action.copy()))
        return infer(*args)

    runtime.policy.infer = observed
    for _ in range(22500):
        runtime.step_physics()
    result = runtime.summary()
    assert result["elapsed"] == pytest.approx(45.)
    assert result["completed_cycles"] == 2
    assert runtime.phase_steps == 2200
    assert [step for step, _ in calls] == [0, 10150, 20300]
    assert all(np.array_equal(action, np.zeros(29)) for _, action in calls)
    assert result["min_z"] > .6 and result["min_up"] > .9
    assert not np.any(runtime.data.eq_active)
    assert capsys.readouterr().out.count("[GANGNAM] Replay cycle") == 2
    runtime.reset()
    assert runtime.steps == runtime.phase_steps == runtime.completed_cycles == 0
    assert runtime.summary()["torque_saturation"] == 0


def test_default_graphical_run_is_unbounded(monkeypatch, viewer_module):
    def viewer(runtime, count):
        assert count is None
        runtime.step_physics()

    monkeypatch.setattr(viewer_module, "run_viewer", viewer, raising=False)
    assert run_gangnam()["steps"] == 1


def test_error_before_replay_boundary_propagates():
    runtime = GangnamRuntime()
    runtime.data.qpos[2] = .3
    with pytest.raises(RuntimeError, match="fell"):
        runtime.step_physics()
    assert runtime.completed_cycles == 0


def test_replay_boundary_does_not_hide_a_fall():
    runtime = GangnamRuntime()
    runtime.phase_steps = round(runtime.policy.duration / runtime.model.opt.timestep)
    runtime.completed_cycles = 1
    runtime.data.qpos[2] = .3
    mujoco.mj_forward(runtime.model, runtime.data)
    with pytest.raises(RuntimeError, match="fell"):
        runtime.step_physics()
    assert runtime.data.qpos[2] == pytest.approx(.3)
    assert runtime.completed_cycles == 1
    assert runtime.steps == 0
