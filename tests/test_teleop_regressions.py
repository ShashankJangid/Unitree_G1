"""Reset and force contracts with controlled input/time, not a real viewer."""
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
from mujoco import viewer
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g1_teleop as teleop


@pytest.fixture
def probe(monkeypatch):
    def run(keys, steps, *, fall=False):
        state = dict(clock=0., resets=0, forces=[], closed=False)
        commands = iter(keys)

        class Policy:
            def eval(self):
                return self

            def reset_memory(self):
                state["resets"] += 1

            def __call__(self, obs):
                return torch.zeros((1, 12))

        class Keyboard:
            def poll(self):
                return next(commands, None)

            def close(self):
                state["closed"] = True

        class Viewer:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def is_running(self):
                return len(state["forces"]) < steps

            def sync(self):
                pass

        def physics(model, data):
            pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            state["forces"].append(float(data.xfrc_applied[pelvis, 0]))
            state["clock"] += model.opt.timestep
            if fall and len(state["forces"]) == 1:
                data.qpos[2] = .3

        monkeypatch.setattr(teleop, "CrossPlatformKeyboard", Keyboard)
        monkeypatch.setattr(teleop, "time", SimpleNamespace(
            time=lambda: state["clock"], sleep=lambda _: None))
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.jit, "load", lambda *args, **kwargs: Policy())
        monkeypatch.setattr(viewer, "launch_passive", lambda *args, **kwargs: Viewer())
        monkeypatch.setattr(mujoco, "mj_step", physics)
        teleop.run_teleop()
        assert state["closed"]
        return state

    return run


@pytest.mark.parametrize("fall", [False, True])
def test_manual_and_fall_resets_clear_policy_memory(probe, fall):
    result = probe([] if fall else ["r"], 2, fall=fall)
    assert result["resets"] == 2  # Initial state plus manual/fall reset.


def test_push_persists_and_expires_at_defined_duration(probe):
    result = probe(["f"], 80)
    force = np.array(result["forces"])
    assert np.count_nonzero(force) > 1
    assert set(force) == {0., 60.}
    assert force[-1] == 0.
    assert force.sum() * .002 == pytest.approx(60 * .15, abs=60 * .002)


def test_manual_reset_clears_pending_force(probe):
    result = probe(["f", "r"], 3)
    assert result["forces"] == [60., 0., 0.]
    assert result["resets"] == 2
