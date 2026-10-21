"""CLI routing and import isolation for display-free demo runs."""

import os
import pathlib
import runpy
import subprocess
import sys
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("entrypoint,mode", [
    *[("main.py", mode) for mode in ["teleop", "walk", "wander", "dance", "wave", "reach", "bow", "punch", "gangnam"]],
    ("g1_teleop.py", ""), ("g1_velocity.py", "walk"), ("g1_expressive.py", "wave"),
])
def test_headless_cli_modes(entrypoint, mode):
    env = os.environ.copy()
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    # Disallow any graphics import even in the presence of a cached package.
    guard = """import builtins, runpy, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    fromlist = (args[2] if len(args) > 2 else kwargs.get('fromlist', ())) or ()
    if name == 'glfw' or name == 'mujoco.viewer' or (name == 'mujoco' and 'viewer' in fromlist):
        raise AssertionError('graphics import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
entrypoint, mode = sys.argv[1:]
sys.argv = [entrypoint] + ([mode] if mode else []) + ['--headless', '--duration', '0.002']
runpy.run_path(entrypoint, run_name='__main__')
"""
    result = subprocess.run([sys.executable, "-c", guard, entrypoint, mode], cwd=ROOT, env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_duration_rejected(value):
    result = subprocess.run([sys.executable, "main.py", "wave", "--headless", f"--duration={value}"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert "positive and finite" in result.stderr


def test_train_duration_rejected_before_training_import():
    result = subprocess.run([sys.executable, "main.py", "train", "--duration", "1"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert "--timesteps" in result.stderr


def test_train_headless_route_without_running_training(monkeypatch):
    calls = []
    training = types.ModuleType("g1_train")
    training.run_training = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "g1_train", training)
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    monkeypatch.setattr(sys, "argv", ["main.py", "train", "--headless", "--timesteps", "7"])
    runpy.run_path(str(ROOT / "main.py"), run_name="__main__")
    assert calls == [{"total_timesteps": 7, "headless": True, "resume_path": None}]
    assert os.environ["MUJOCO_GL"] == "disable"
