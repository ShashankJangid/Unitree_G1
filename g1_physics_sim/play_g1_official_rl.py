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
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from g1_constants import NOMINAL_STAND_Q, OFFICIAL_KP, OFFICIAL_KD, TORQUE_LIMITS

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
    print(f"[OFFICIAL RL] Loading Unitree G1 Full-Motion RL Controller on {device.upper()}...")

    policy_file = base_dir / "models" / "motion.pt"
    if not policy_file.exists():
        policy_file = gym_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    if not policy_file.exists():
        raise FileNotFoundError(f"motion.pt not found at {policy_file}")

    policy = torch.jit.load(str(policy_file), map_location=device)
    policy.eval()

    xml_file = base_dir / "models" / "g1" / "g1_physics_scene.xml"
    if not xml_file.exists():
        xml_file = gym_root / "resources" / "robots" / "g1_description" / "scene.xml"

    m = mujoco.MjModel.from_xml_path(str(xml_file))
    d = mujoco.MjData(m)
    m.opt.timestep = 0.002

    pelvis_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")

    kps_legs = np.array([100, 100, 100, 150, 40, 40, 100, 100, 100, 150, 40, 40], dtype=np.float32)
    kds_legs = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2], dtype=np.float32)
    default_angles = np.array([-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0], dtype=np.float32)

    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    action_scale = 0.25
    cmd_scale = np.array([2.0, 2.0, 0.25], dtype=np.float32)

    num_actions = 12
    num_obs = 47
    control_decimation = 10

    target_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    cur_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    is_29dof = m.nu >= 29

    arm_poses = {
        "natural": NOMINAL_STAND_Q[12:29].copy(),
        "ready": np.array([
            0.0, 0.0, 0.0,
            0.6, 0.2, 0.0, 1.2, 0.0, 0.0, 0.0,
            0.6, -0.2, 0.0, 1.2, 0.0, 0.0, 0.0
        ], dtype=np.float32),
        "wave": np.array([
            0.0, 0.0, 0.0,
            0.3, 0.25, 0.0, 0.8, 0.0, 0.0, 0.0,
            -0.8, -0.3, 0.4, 1.4, 0.0, 0.0, 0.0
        ], dtype=np.float32),
        "tpose": np.array([
            0.0, 0.0, 0.0,
            0.0, 1.4, 0.0, 0.2, 0.0, 0.0, 0.0,
            0.0, -1.4, 0.0, 0.2, 0.0, 0.0, 0.0
        ], dtype=np.float32)
    }
    cur_arm_target = arm_poses["natural"].copy()

    def reset_robot():
        nonlocal action, target_dof_pos, obs, target_cmd, cur_cmd
        mujoco.mj_resetData(m, d)
        d.qpos[0] = 0.0
        d.qpos[1] = 0.0
        d.qpos[2] = 0.793
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qpos[7:19] = default_angles
        if is_29dof:
            d.qpos[19:36] = cur_arm_target
        d.qvel[:] = 0.0
        action = np.zeros(num_actions, dtype=np.float32)
        target_dof_pos = default_angles.copy()
        obs = np.zeros(num_obs, dtype=np.float32)
        target_cmd[:] = 0.0
        cur_cmd[:] = 0.0
        mujoco.mj_forward(m, d)

    reset_robot()

    def apply_key_action(action_key: str):
        nonlocal target_cmd, cur_arm_target
        k = action_key.lower()
        if k in ('w', 'up'):
            target_cmd[0] = float(np.clip(target_cmd[0] + 0.15, -0.6, 1.2))
        elif k in ('s', 'down'):
            target_cmd[0] = float(np.clip(target_cmd[0] - 0.15, -0.6, 1.2))
        elif k in ('a', 'left'):
            target_cmd[2] = float(np.clip(target_cmd[2] + 0.25, -1.2, 1.2))
        elif k in ('d', 'right'):
            target_cmd[2] = float(np.clip(target_cmd[2] - 0.25, -1.2, 1.2))
        elif k == 'q':
            target_cmd[1] = float(np.clip(target_cmd[1] + 0.15, -0.6, 0.6))
        elif k == 'e':
            target_cmd[1] = float(np.clip(target_cmd[1] - 0.15, -0.6, 0.6))
        elif k in (' ', 'x'):
            target_cmd[:] = 0.0
        elif k == 'r':
            print("[ACTION] Manual Stance Reset Triggered.")
            reset_robot()
            return
        elif k == 'f':
            d.xfrc_applied[pelvis_bid, 0] = 60.0
            print("[PERTURBATION] >>> Applied 60 N physical push to torso! <<<")
            return
        elif k == '1':
            cur_arm_target = arm_poses["natural"].copy()
            print("[ARM POSE] Set Upper Body -> Natural Walking Stance")
            return
        elif k == '2':
            cur_arm_target = arm_poses["ready"].copy()
            print("[ARM POSE] Set Upper Body -> Forward Ready / Guard")
            return
        elif k == '3':
            cur_arm_target = arm_poses["wave"].copy()
            print("[ARM POSE] Set Upper Body -> Right Arm Wave")
            return
        elif k == '4':
            cur_arm_target = arm_poses["tpose"].copy()
            print("[ARM POSE] Set Upper Body -> T-Pose Spread")
            return

        print(f"[CMD] vx = {target_cmd[0]:+.2f} m/s | vy = {target_cmd[1]:+.2f} m/s | wz = {target_cmd[2]:+.2f} rad/s")

    def viewer_key_callback(keycode: int):
        if keycode in (ord('W'), ord('w'), 265):
            apply_key_action('w')
        elif keycode in (ord('S'), ord('s'), 264):
            apply_key_action('s')
        elif keycode in (ord('A'), ord('a'), 263):
            apply_key_action('a')
        elif keycode in (ord('D'), ord('d'), 262):
            apply_key_action('d')
        elif keycode in (ord('Q'), ord('q')):
            apply_key_action('q')
        elif keycode in (ord('E'), ord('e')):
            apply_key_action('e')
        elif keycode in (ord(' '), ord('X'), ord('x')):
            apply_key_action(' ')
        elif keycode in (ord('R'), ord('r')):
            apply_key_action('r')
        elif keycode in (ord('F'), ord('f')):
            apply_key_action('f')
        elif keycode == ord('1'):
            apply_key_action('1')
        elif keycode == ord('2'):
            apply_key_action('2')
        elif keycode == ord('3'):
            apply_key_action('3')
        elif keycode == ord('4'):
            apply_key_action('4')

    print("\n" + "=" * 76)
    print(">>> 29-DOF FULL-MOTION INTERACTIVE RL TELEOPERATION RUNNER <<<")
    print("=" * 76)
    print("  Controls work both in the 3D MuJoCo Window AND in this Terminal:")
    print("    W / Up Arrow    : Forward (+0.15 m/s)")
    print("    S / Down Arrow  : Backward / Decelerate (-0.15 m/s)")
    print("    A / Left Arrow  : Turn Left / Yaw (+0.25 rad/s)")
    print("    D / Right Arrow : Turn Right / Yaw (-0.25 rad/s)")
    print("    Q / E           : Strafe Left / Strafe Right (+/- 0.15 m/s)")
    print("    Space / X       : Full Stop / Dynamic Stand In Place")
    print("    1 / 2 / 3 / 4   : Arm Postures (Natural / Ready / Wave / T-Pose)")
    print("    R               : Instant Reset Stance")
    print("    F               : Apply Torso Perturbation Impulse (60 N)")
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
                        apply_key_action('w')
                    elif ch2 == b'P':
                        apply_key_action('s')
                    elif ch2 == b'K':
                        apply_key_action('a')
                    elif ch2 == b'M':
                        apply_key_action('d')
                else:
                    k = ch.decode('utf-8', errors='ignore')
                    apply_key_action(k)

            cur_cmd += 0.08 * (target_cmd - cur_cmd)

            tau_legs = (target_dof_pos - d.qpos[7:19]) * kps_legs - d.qvel[6:18] * kds_legs
            d.ctrl[:12] = np.clip(tau_legs, -TORQUE_LIMITS[:12], TORQUE_LIMITS[:12])

            if is_29dof:
                tau_upper = (cur_arm_target - d.qpos[19:36]) * OFFICIAL_KP[12:29] - d.qvel[18:35] * OFFICIAL_KD[12:29]
                d.ctrl[12:29] = np.clip(tau_upper, -TORQUE_LIMITS[12:29], TORQUE_LIMITS[12:29])

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
