from __future__ import annotations
import math
import argparse
import os
import pathlib
import sys
import time
import torch
import numpy as np

if __name__ == "__main__" and "--headless" in sys.argv[1:]:
    os.environ["MUJOCO_GL"] = "disable"

if sys.platform.startswith("linux") and os.environ.get("DISPLAY") and "PYGLFW_LIBRARY_VARIANT" not in os.environ:
    os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"

import mujoco

root_dir = pathlib.Path(__file__).resolve().parent
xml_file = root_dir / "models" / "g1" / "scene.xml"
policy_file = root_dir / "models" / "motion.pt"


class CrossPlatformKeyboard:
    """Non-blocking keyboard reader supporting Windows, Linux, and macOS."""

    def __init__(self):
        self.is_windows = sys.platform == "win32"
        self.old_settings = None
        if not self.is_windows and sys.stdin.isatty():
            try:
                import termios
                import tty
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                self.old_settings = None

    def poll(self) -> str | None:
        if self.is_windows:
            try:
                import msvcrt
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    if ch in (b'\x00', b'\xe0'):
                        ch2 = msvcrt.getch()
                        mapping = {b'H': 'w', b'P': 's', b'K': 'a', b'M': 'd'}
                        return mapping.get(ch2)
                    return ch.decode('utf-8', errors='ignore')
            except Exception:
                return None
        else:
            if self.old_settings is not None:
                import select
                try:
                    r, _, _ = select.select([sys.stdin], [], [], 0)
                    if r:
                        ch = sys.stdin.read(1)
                        if ch == '\x1b':
                            r2, _, _ = select.select([sys.stdin], [], [], 0.01)
                            if r2:
                                seq = sys.stdin.read(2)
                                if seq == '[A': return 'w'
                                elif seq == '[B': return 's'
                                elif seq == '[D': return 'a'
                                elif seq == '[C': return 'd'
                        return ch
                except Exception:
                    return None
        return None

    def close(self):
        if not self.is_windows and self.old_settings is not None:
            try:
                import termios
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz)
    ], dtype=np.float32)


def run_teleop(duration: float | None = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[G1 TELEOP] Starting Reactive Locomotion Controller on {device.upper()}...")

    if not policy_file.exists():
        raise FileNotFoundError(f"Policy file not found: {policy_file}")
    if not xml_file.exists():
        raise FileNotFoundError(f"Scene XML not found: {xml_file}")

    policy = torch.jit.load(str(policy_file), map_location=device)
    policy.eval()

    m = mujoco.MjModel.from_xml_path(str(xml_file))
    d = mujoco.MjData(m)
    m.opt.timestep = 0.002

    pelvis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    kps = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
    kds = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
    default_angles = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)

    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    action_scale = 0.25
    cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)

    num_actions = 12
    num_obs = 47
    control_decimation = 10

    base_cruise_vx = 0.40
    burst_vx = 0.40
    burst_vy = 0.00
    burst_wz = 0.00

    expire_vx = 0.0
    expire_vy = 0.0
    expire_wz = 0.0
    perturb_expire = 0.0

    target_cmd = np.array([0.40, 0.0, 0.0], dtype=np.float32)
    cur_cmd = np.array([0.40, 0.0, 0.0], dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    def reset_robot():
        nonlocal action, target_dof_pos, obs, target_cmd, cur_cmd
        nonlocal burst_vx, burst_vy, burst_wz, expire_vx, expire_vy, expire_wz, perturb_expire
        mujoco.mj_resetData(m, d)
        d.qpos[0] = 0.0
        d.qpos[1] = 0.0
        d.qpos[2] = 0.793
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[7:19] = default_angles
        d.qvel[:] = 0.0
        d.xfrc_applied[:] = 0.0

        action = np.zeros(num_actions, dtype=np.float32)
        target_dof_pos = default_angles.copy()
        obs = np.zeros(num_obs, dtype=np.float32)
        burst_vx = base_cruise_vx
        burst_vy = 0.0
        burst_wz = 0.0
        expire_vx = 0.0
        expire_vy = 0.0
        expire_wz = 0.0
        perturb_expire = 0.0
        target_cmd[:] = [base_cruise_vx, 0.0, 0.0]
        cur_cmd[:] = [base_cruise_vx, 0.0, 0.0]

        # Reset recurrent LSTM state in TorchScript policy
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

        mujoco.mj_forward(m, d)

    reset_robot()

    def handle_reactive_input(k_str: str):
        nonlocal burst_vx, burst_vy, burst_wz, expire_vx, expire_vy, expire_wz, perturb_expire
        now = time.time()
        k = k_str.lower()
        if k in ('w', 'up'):
            burst_vx = 1.00
            expire_vx = now + 0.65
            print("[BURST] >> SPRINT FORWARD (1.00 m/s)")
        elif k == 'z':
            burst_vx = 1.40
            expire_vx = now + 0.70
            print("[BURST] >>> TURBO SPRINT (1.40 m/s) <<<")
        elif k in ('s', 'down'):
            burst_vx = -0.35
            expire_vx = now + 0.60
            print("[BURST] << ACTIVE BRAKE / REVERSE (-0.35 m/s)")
        elif k in ('a', 'left'):
            burst_wz = 0.85
            expire_wz = now + 0.50
            print("[BURST] <-- SHARP LEFT TURN (+0.85 rad/s)")
        elif k in ('d', 'right'):
            burst_wz = -0.85
            expire_wz = now + 0.50
            print("[BURST] -- SHARP RIGHT TURN (-0.85 rad/s) -->")
        elif k == 'q':
            burst_vy = 0.45
            expire_vy = now + 0.50
            print("[BURST] <<-- STRAFE LEFT (+0.45 m/s)")
        elif k == 'e':
            burst_vy = -0.45
            expire_vy = now + 0.50
            print("[BURST] -- STRAFE RIGHT (-0.45 m/s) -->>")
        elif k in (' ', 'x'):
            burst_vx = base_cruise_vx
            burst_vy = 0.0
            burst_wz = 0.0
            expire_vx = 0.0
            expire_vy = 0.0
            expire_wz = 0.0
            print(f"[CMD] Steady Cruise ({base_cruise_vx:.2f} m/s)")
        elif k == 'r':
            print("[ACTION] Instant Robot Reset.")
            reset_robot()
        elif k == 'f':
            perturb_expire = now + 0.15  # Sustained 150ms perturbation
            print("[PERTURBATION] Applied 60 N torso push impulse (150ms).")

    def viewer_key_callback(keycode: int):
        if keycode in (ord('W'), ord('w'), 265):
            handle_reactive_input('w')
        elif keycode in (ord('Z'), ord('z')):
            handle_reactive_input('z')
        elif keycode in (ord('S'), ord('s'), 264):
            handle_reactive_input('s')
        elif keycode in (ord('A'), ord('a'), 263):
            handle_reactive_input('a')
        elif keycode in (ord('D'), ord('d'), 262):
            handle_reactive_input('d')
        elif keycode in (ord('Q'), ord('q')):
            handle_reactive_input('q')
        elif keycode in (ord('E'), ord('e')):
            handle_reactive_input('e')
        elif keycode in (ord(' '), ord('X'), ord('x')):
            handle_reactive_input(' ')
        elif keycode in (ord('R'), ord('r')):
            handle_reactive_input('r')
        elif keycode in (ord('F'), ord('f')):
            handle_reactive_input('f')

    print("\n" + "=" * 76)
    print(">>> UNITREE G1 REACTIVE OMNIDIRECTIONAL TELEOPERATION <<<")
    print("=" * 76)
    print("  Controls work both in the 3D MuJoCo Window AND in this Terminal:")
    print("    W / Up Arrow    : Fast Sprint Burst (1.00 m/s)")
    print("    Z               : Turbo Max Sprint (1.40 m/s)")
    print("    S / Down Arrow  : Active Decel / Brake (-0.35 m/s)")
    print("    A / Left Arrow  : Sharp Left Turn (+0.85 rad/s)")
    print("    D / Right Arrow : Sharp Right Turn (-0.85 rad/s)")
    print("    Q / E           : Strafe Left / Right (+/- 0.45 m/s)")
    print("    Space / X       : Return to Steady Cruise (0.40 m/s)")
    print("    R               : Reset Robot Stance & Memory")
    print("    F               : Apply 60 N Push Perturbation")
    print("=" * 76 + "\n")

    counter = 0
    total_steps = 0
    episode = 1
    last_print = time.time()
    last_render = time.time()
    render_interval = 1.0 / 60.0  # Smooth 60 FPS rendering

    kbd = CrossPlatformKeyboard()

    try:
        from mujoco import viewer as mj_viewer
        with mj_viewer.launch_passive(m, d, key_callback=viewer_key_callback) as viewer:
            while viewer.is_running():
                if duration is not None and total_steps * m.opt.timestep >= duration:
                    break
                step_start = time.time()
                now = time.time()

                # Poll cross-platform keyboard
                term_key = kbd.poll()
                if term_key:
                    handle_reactive_input(term_key)

                t_vx = burst_vx if now < expire_vx else base_cruise_vx
                t_vy = burst_vy if now < expire_vy else 0.00
                t_wz = burst_wz if now < expire_wz else 0.00
                target_cmd[:] = [t_vx, t_vy, t_wz]

                cur_cmd += 0.12 * (target_cmd - cur_cmd)

                tau = (target_dof_pos - d.qpos[7:19]) * kps - d.qvel[6:18] * kds
                d.ctrl[:12] = tau

                # Sustained push perturbation
                if now < perturb_expire and pelvis_bid != -1:
                    d.xfrc_applied[pelvis_bid, 0] = 60.0
                else:
                    d.xfrc_applied[pelvis_bid, :] = 0.0

                mujoco.mj_step(m, d)

                counter += 1
                total_steps += 1
                if counter % control_decimation == 0:
                    qj = (d.qpos[7:19] - default_angles) * dof_pos_scale
                    dqj = d.qvel[6:18] * dof_vel_scale
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6] * ang_vel_scale

                    period = 0.8
                    count = counter * 0.002
                    phase = (count % period) / period
                    sin_phase = np.sin(2.0 * np.pi * phase)
                    cos_phase = np.cos(2.0 * np.pi * phase)

                    obs[:3] = omega
                    obs[3:6] = get_gravity_orientation(quat)
                    obs[6:9] = cur_cmd * cmd_scale
                    obs[9:21] = qj
                    obs[21:33] = dqj
                    obs[33:45] = action
                    obs[45:47] = np.array([sin_phase, cos_phase], dtype=np.float32)

                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action = policy(obs_t).squeeze().cpu().numpy()
                    target_dof_pos = action * action_scale + default_angles

                z = float(d.qpos[2])
                R = d.xmat[pelvis_bid].reshape(3, 3) if pelvis_bid != -1 else np.eye(3)
                up_z = float(R[2, 2])

                if z < 0.45 or up_z < 0.65 or not np.isfinite(z):
                    reason = "fallen_too_low" if z < 0.45 else "tilted_too_much"
                    print(f"[AUTO-RESET] Episode {episode}: Triggered reset ({reason}, z={z:.2f}m, up_z={up_z:.2f})")
                    episode += 1
                    reset_robot()
                    counter = 0

                # 60 FPS throttled viewer sync
                if now - last_render >= render_interval:
                    viewer.sync()
                    last_render = now

                if time.time() - last_print >= 0.5:
                    print(
                        f"Step: {counter:5d} | Height: {z:.2f}m | "
                        f"Speed: vx={d.qvel[0]:+.2f} m/s, vy={d.qvel[1]:+.2f} m/s, wz={d.qvel[5]:+.2f} rad/s | "
                        f"Cmd: ({cur_cmd[0]:.2f}, {cur_cmd[1]:.2f}, {cur_cmd[2]:.2f})"
                    )
                    last_print = time.time()

                elapsed = time.time() - step_start
                sleep_time = m.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n[G1 TELEOP] Simulation stopped cleanly by user.")
    finally:
        kbd.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unitree G1 teleoperation")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")
    if args.headless:
        from g1_velocity import run_velocity_demo
        run_velocity_demo(mode="walk", headless=True, duration=args.duration)
    else:
        run_teleop(duration=args.duration)
