from __future__ import annotations
import math
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


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OFFICIAL RL] Loading Unitree G1 Official Policy on {device.upper()}...")

    policy_file = base_dir / "models" / "motion.pt"
    if not policy_file.exists():
        policy_file = gym_root / "deploy" / "pre_train" / "g1" / "motion.pt"
    if not policy_file.exists():
        raise FileNotFoundError(f"motion.pt not found at {policy_file}")

    xml_file = gym_root / "resources" / "robots" / "g1_description" / "scene.xml"
    if not xml_file.exists():
        xml_file = base_dir / "models" / "g1" / "g1_physics_scene.xml"

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

    cmd = np.array([0.4, 0.0, 0.0], dtype=np.float32)
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    def reset_robot():
        nonlocal action, target_dof_pos, obs
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
        mujoco.mj_forward(m, d)

    reset_robot()

    print("\n" + "=" * 65)
    print(">>> OFFICIAL UNITREE G1 RL POLICY VIEWER (PRE-TRAINED CHECKPOINT) <<<")
    print(">>> Interactive 3D MuJoCo window active. Robot balances & walks via RL!")
    print(">>> Controls: W/S (Speed +/-), A/D (Turn L/R), Space/X (Stop), R (Reset)")
    print("=" * 65 + "\n")

    counter = 0
    episode = 1
    last_print = time.time()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            step_start = time.time()

            tau = pd_control(target_dof_pos, d.qpos[7:19], kps, np.zeros_like(kds), d.qvel[6:18], kds)
            d.ctrl[:12] = tau
            mujoco.mj_step(m, d)

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
                obs[6:9] = cmd * cmd_scale
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

            if time.time() - last_print >= 1.0:
                print(
                    f"Episode {episode} | Step: {counter:5d} | Height: {z:.2f}m | "
                    f"Fwd Vel: {d.qvel[0]:+.2f} m/s | Cmd: vx={cmd[0]:.2f}, wz={cmd[2]:.2f}"
                )
                last_print = time.time()

            elapsed = time.time() - step_start
            sleep_time = m.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
