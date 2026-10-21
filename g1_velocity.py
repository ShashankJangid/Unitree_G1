"""Reliable commanded/random walking and push recovery for the G1.

The existing verified 12-DoF TorchScript locomotion checkpoint is paired with
the matching 12-DoF MuJoCo model.  The official 29-DoF Unitree RL Lab artifact
is kept under ``models/unitree_velocity`` for later model-matched integration;
it is not silently used here because this repository's custom 29-DoF XML does
not remain upright with that checkpoint when the gantry is disabled.
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import time

if __name__ == "__main__" and "--headless" in sys.argv[1:]:
    os.environ["MUJOCO_GL"] = "disable"

import mujoco
import numpy as np
import torch

from g1_teleop import CrossPlatformKeyboard, get_gravity_orientation


ROOT = pathlib.Path(__file__).resolve().parent
SCENE = ROOT / "models/g1/scene.xml"
POLICY = ROOT / "models/motion.pt"


def run_velocity_demo(
    mode: str = "wander",
    headless: bool = False,
    push_force: float = 120.0,
    auto_push: bool = False,
    duration: float | None = None,
):
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be positive and finite")
    if not SCENE.is_file() or not POLICY.is_file():
        raise FileNotFoundError(f"Missing walking assets: {SCENE} / {POLICY}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = torch.jit.load(str(POLICY), map_location=device)
    policy.eval()

    def reset_policy_memory():
        if hasattr(policy, "reset_memory"):
            try:
                policy.reset_memory()
            except Exception:
                pass
        elif hasattr(policy, "hidden_state"):
            try:
                policy.hidden_state.zero_()
                policy.cell_state.zero_()
            except Exception:
                pass
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    kps = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
    kds = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
    default_angles = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)
    cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)
    action_scale = 0.25
    target_cmd = np.zeros(3, dtype=np.float32)
    cur_cmd = np.zeros(3, dtype=np.float32)
    action = np.zeros(12, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(47, dtype=np.float32)
    rng = np.random.default_rng(20261019)
    push_until = 0.0
    push_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim_time = 0.0
    step_count = 0
    next_wander = 0.0
    push_fired = False
    last_print = time.time()
    last_render = time.time()
    keyboard = None if headless else CrossPlatformKeyboard()
    total_steps = 0
    resets = 0

    def reset():
        nonlocal action, target_dof_pos, cur_cmd, target_cmd, sim_time, step_count
        mujoco.mj_resetData(model, data)
        data.qpos[:7] = [0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0]
        data.qpos[7:19] = default_angles
        data.qvel[:] = 0.0
        data.xfrc_applied[:] = 0.0
        action = np.zeros(12, dtype=np.float32)
        target_dof_pos = default_angles.copy()
        # This checkpoint was trained with a small forward cruise command;
        # zero velocity is a crouch/stop behavior rather than its nominal gait.
        cur_cmd[:] = [0.4, 0.0, 0.0]
        target_cmd[:] = [0.4, 0.0, 0.0]
        reset_policy_memory()
        mujoco.mj_forward(model, data)

    def handle_key(key: str):
        nonlocal push_until, push_direction
        key = key.lower()
        if key == "w":
            target_cmd[:] = [1.0, 0.0, 0.0]
        elif key == "s":
            target_cmd[:] = [-0.35, 0.0, 0.0]
        elif key == "a":
            target_cmd[:] = [0.0, 0.45, 0.0]
        elif key == "d":
            target_cmd[:] = [0.0, -0.45, 0.0]
        elif key == "q":
            target_cmd[:] = [0.0, 0.0, 0.85]
        elif key == "e":
            target_cmd[:] = [0.0, 0.0, -0.85]
        elif key in (" ", "x"):
            target_cmd[:] = 0.0
        elif key == "r":
            reset()
        elif key == "f":
            push_direction = rng.normal(size=3).astype(np.float32)
            push_direction[2] = 0.0
            push_direction /= max(float(np.linalg.norm(push_direction)), 1e-6)
            push_until = sim_time + 0.25
            print(f"[PUSH] {push_direction * push_force} N for 250 ms")

    def randomize_command():
        target_cmd[:] = [rng.uniform(-0.30, 0.85), rng.uniform(-0.25, 0.25), rng.uniform(-0.55, 0.55)]

    reset()
    print(f"[VELOCITY] Verified 12-DoF locomotion policy on {device.upper()}")
    print(f"[VELOCITY] Mode={mode.upper()} | push={push_force:.0f} N | gantry=DISABLED")
    print("[VELOCITY] W/A/S/D strafe, Q/E turn, Space stop, F strong push, R reset")

    def loop(viewer=None):
        nonlocal sim_time, step_count, next_wander, last_print, last_render, cur_cmd, target_dof_pos, action, push_fired, total_steps, resets
        budget = math.ceil((duration if duration is not None else 10.0) / model.opt.timestep) if headless else None
        while (total_steps < budget if headless else viewer.is_running()):
            if not headless and duration is not None and total_steps * model.opt.timestep >= duration:
                break
            started = time.time()
            term_key = keyboard.poll() if keyboard is not None else None
            if term_key:
                handle_key(term_key)
            if mode == "wander" and sim_time >= next_wander:
                randomize_command()
                next_wander = sim_time + 2.5
            if auto_push and not push_fired and sim_time >= 3.0:
                push_fired = True
                handle_key("f")
            cur_cmd += 0.12 * (target_cmd - cur_cmd)
            # Match the checkpoint's original deployment path.  These
            # actuators are not ctrl-limited in the XML; pre-clipping the
            # torque here changes the learned gait and causes a collapse.
            torque = kps * (target_dof_pos - data.qpos[7:19]) - kds * data.qvel[6:18]
            if not np.all(np.isfinite(torque)):
                raise FloatingPointError("Non-finite walking torque")
            data.ctrl[:12] = torque
            if sim_time < push_until:
                if not math.isfinite(push_force):
                    raise FloatingPointError("Non-finite push force")
                data.xfrc_applied[pelvis_id, :3] = push_direction * push_force
            else:
                data.xfrc_applied[pelvis_id, :] = 0.0
            mujoco.mj_step(model, data)
            sim_time += model.opt.timestep
            step_count += 1
            total_steps += 1
            if step_count % 10 == 0:
                obs[:3] = data.qvel[3:6] * 0.25
                obs[3:6] = get_gravity_orientation(data.qpos[3:7])
                obs[6:9] = cur_cmd * cmd_scale
                obs[9:21] = data.qpos[7:19] - default_angles
                obs[21:33] = data.qvel[6:18] * 0.05
                obs[33:45] = action
                phase = (step_count * model.opt.timestep % 0.8) / 0.8
                obs[45:47] = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
                with torch.no_grad():
                    action = policy(torch.from_numpy(obs).unsqueeze(0).to(device)).squeeze().cpu().numpy()
                if not np.all(np.isfinite(action)):
                    raise FloatingPointError("Non-finite walking policy action")
                target_dof_pos = action * action_scale + default_angles
            pelvis_up = float(data.xmat[pelvis_id].reshape(3, 3)[2, 2])
            if data.qpos[2] < 0.45 or pelvis_up < 0.65 or not np.isfinite(data.qpos[2]):
                print(f"[AUTO-RESET] fall at t={sim_time:.2f}s z={data.qpos[2]:.2f} up_z={pelvis_up:.2f}")
                resets += 1
                if headless:
                    raise RuntimeError(f"Walking fell after {total_steps} steps; resets={resets}")
                reset()
                next_wander = sim_time + 0.5
            if time.time() - last_print >= 1.0:
                print(f"[VELOCITY] t={sim_time:5.1f}s cmd=({cur_cmd[0]:+.2f},{cur_cmd[1]:+.2f},{cur_cmd[2]:+.2f}) z={data.qpos[2]:.2f} up_z={pelvis_up:.2f}")
                last_print = time.time()
            if viewer is not None and time.time() - last_render >= 1.0 / 60.0:
                viewer.sync()
                last_render = time.time()
            if not headless:
                remaining = model.opt.timestep - (time.time() - started)
                if remaining > 0:
                    time.sleep(remaining)

    try:
        if headless:
            loop()
            print(f"[HEADLESS] Completed {total_steps} steps; resets={resets}")
        else:
            from mujoco import viewer as mj_viewer
            with mj_viewer.launch_passive(model, data, key_callback=lambda key: handle_key(chr(key)) if 0 <= key < 256 else None) as viewer:
                loop(viewer)
    finally:
        if keyboard is not None:
            keyboard.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unitree G1 walking and push recovery")
    parser.add_argument("mode", nargs="?", default="wander", choices=["walk", "wander"])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--push-force", type=float, default=120.0)
    parser.add_argument("--auto-push", action="store_true", help="Apply one random push at simulated t=3s")
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")
    run_velocity_demo(args.mode, args.headless, args.push_force, args.auto_push, args.duration)
