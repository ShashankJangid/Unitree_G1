from __future__ import annotations
import math
import msvcrt
import os
import pathlib
import sys
import time
import torch
import numpy as np
import mujoco
import mujoco.viewer

base_dir = pathlib.Path(__file__).resolve().parent
gym_root = base_dir.parent / "official_unitree_repos" / "unitree_rl_gym"


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    return np.array([
        2.0 * (-qz * qx + qw * qy),
        -2.0 * (qz * qy + qw * qx),
        1.0 - 2.0 * (qw * qw + qz * qz)
    ], dtype=np.float32)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OFFICIAL RL] Launching Official Unitree G1 Controller on {device.upper()}...")

    policy_file = gym_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    if not policy_file.exists():
        policy_file = base_dir / "models" / "motion.pt"
    if not policy_file.exists():
        raise FileNotFoundError(f"motion.pt not found at {policy_file}")

    xml_file = gym_root / "resources" / "robots" / "g1_description" / "scene.xml"
    if not xml_file.exists():
        xml_file = base_dir / "models" / "g1" / "scene.xml"
    if not xml_file.exists():
        raise FileNotFoundError(f"scene.xml not found at {xml_file}")

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

    target_cmd = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    cur_cmd = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    def reset_robot():
        nonlocal action, target_dof_pos, obs, target_cmd, cur_cmd
        mujoco.mj_resetData(m, d)
        d.qpos[0] = 0.0
        d.qpos[1] = 0.0
        d.qpos[2] = 0.793
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[7:19] = default_angles
        d.qvel[:] = 0.0
        action = np.zeros(num_actions, dtype=np.float32)
        target_dof_pos = default_angles.copy()
        obs = np.zeros(num_obs, dtype=np.float32)
        target_cmd[:] = [0.4, 0.0, 0.0]
        cur_cmd[:] = [0.4, 0.0, 0.0]
        mujoco.mj_forward(m, d)

    reset_robot()

    def apply_command(k_str: str):
        nonlocal target_cmd
        k = k_str.lower()
        if k in ('w', 'up'):
            target_cmd[0] = float(np.clip(target_cmd[0] + 0.1, -0.4, 1.2))
        elif k in ('s', 'down'):
            target_cmd[0] = float(np.clip(target_cmd[0] - 0.1, -0.4, 1.2))
        elif k in ('a', 'left'):
            target_cmd[2] = float(np.clip(target_cmd[2] + 0.2, -1.0, 1.0))
        elif k in ('d', 'right'):
            target_cmd[2] = float(np.clip(target_cmd[2] - 0.2, -1.0, 1.0))
        elif k == 'q':
            target_cmd[1] = float(np.clip(target_cmd[1] + 0.1, -0.4, 0.4))
        elif k == 'e':
            target_cmd[1] = float(np.clip(target_cmd[1] - 0.1, -0.4, 0.4))
        elif k in (' ', 'x'):
            target_cmd[:] = [0.4, 0.0, 0.0]
            print("[CMD] Cruising Walk reset to 0.40 m/s")
        elif k == 'r':
            print("[ACTION] Manual Reset Triggered.")
            reset_robot()
            return
        elif k == 'f':
            d.xfrc_applied[pelvis_bid, 0] = 50.0
            print("[PERTURBATION] >>> Applied 50 N physical push to torso! <<<")
            return

        print(f"[CMD] vx = {target_cmd[0]:+.2f} m/s | vy = {target_cmd[1]:+.2f} m/s | wz = {target_cmd[2]:+.2f} rad/s")

    def viewer_key_callback(keycode: int):
        if keycode in (ord('W'), ord('w'), 265):
            apply_command('w')
        elif keycode in (ord('S'), ord('s'), 264):
            apply_command('s')
        elif keycode in (ord('A'), ord('a'), 263):
            apply_command('a')
        elif keycode in (ord('D'), ord('d'), 262):
            apply_command('d')
        elif keycode in (ord('Q'), ord('q')):
            apply_command('q')
        elif keycode in (ord('E'), ord('e')):
            apply_command('e')
        elif keycode in (ord(' '), ord('X'), ord('x')):
            apply_command(' ')
        elif keycode in (ord('R'), ord('r')):
            apply_command('r')
        elif keycode in (ord('F'), ord('f')):
            apply_command('f')

    print("\n" + "=" * 76)
    print(">>> OFFICIAL UNITREE G1 OMNIDIRECTIONAL LOCOMOTION TELEOP <<<")
    print("=" * 76)
    print("  Controls work both inside the MuJoCo 3D Window AND in this Terminal:")
    print("    W / Up Arrow    : Accelerate Forward (+0.10 m/s)")
    print("    S / Down Arrow  : Decelerate / Reverse (-0.10 m/s)")
    print("    A / Left Arrow  : Turn Left (+0.20 rad/s)")
    print("    D / Right Arrow : Turn Right (-0.20 rad/s)")
    print("    Q / E           : Strafe Left / Right (+/- 0.10 m/s)")
    print("    Space / X       : Reset to Steady Cruising Walk (0.40 m/s)")
    print("    R               : Instant Reset Robot Stance")
    print("    F               : Torso Push Perturbation (50 N)")
    print("=" * 76 + "\n")

    counter = 0
    episode = 1
    last_print = time.time()

    with mujoco.viewer.launch_passive(m, d, key_callback=viewer_key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b'\x00', b'\xe0'):
                    ch2 = msvcrt.getch()
                    if ch2 == b'H':
                        apply_command('w')
                    elif ch2 == b'P':
                        apply_command('s')
                    elif ch2 == b'K':
                        apply_command('a')
                    elif ch2 == b'M':
                        apply_command('d')
                else:
                    k = ch.decode('utf-8', errors='ignore')
                    apply_command(k)

            cur_cmd += 0.05 * (target_cmd - cur_cmd)

            tau = (target_dof_pos - d.qpos[7:19]) * kps - d.qvel[6:18] * kds
            d.ctrl[:12] = tau

            mujoco.mj_step(m, d)
            d.xfrc_applied[pelvis_bid, :] = 0.0

            counter += 1
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

            viewer.sync()

            if time.time() - last_print >= 0.5:
                print(
                    f"Step: {counter:5d} | Height: {z:.2f}m | "
                    f"Vel: vx={d.qvel[0]:+.2f}, vy={d.qvel[1]:+.2f}, wz={d.qvel[5]:+.2f} | "
                    f"Cmd: ({cur_cmd[0]:.2f}, {cur_cmd[1]:.2f}, {cur_cmd[2]:.2f})"
                )
                last_print = time.time()

            elapsed = time.time() - step_start
            sleep_time = m.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
