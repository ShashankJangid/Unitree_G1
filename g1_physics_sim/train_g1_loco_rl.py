from __future__ import annotations
import argparse
import os
import pathlib
import sys
import time
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

base_dir = pathlib.Path(__file__).resolve().parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from g1_loco_rl_env import G1LocomotionEnv


class LiveConsoleCallback(BaseCallback):
    def __init__(self, check_freq: int = 100):
        super().__init__()
        self.check_freq = check_freq
        self.episode_count = 0
        self.ep_rewards = []
        self.ep_lengths = []
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        for info in infos:
            if 'step' in info and self.locals.get('dones', [False])[0]:
                self.episode_count += 1
                ep_reward = sum(self.locals.get('rewards', [0.0]))
                reason = info.get('reason', 'completed')
                z = info.get('z', 0.0)
                vx = info.get('vx', 0.0)
                step = info.get('step', 0)

                status = f"[FALL RESET: {reason}]" if reason else "[TRUNCATED]"
                print(
                    f"Episode {self.episode_count:4d} | Steps: {step:3d} | "
                    f"Height: {z:.2f}m | Fwd Vel: {vx:+.2f}m/s | {status}"
                )

        if self.n_calls % self.check_freq == 0:
            fps = int(self.model.num_timesteps / max(1e-3, (time.time() - self.model.start_time)))
            print(
                f">>> Total Timesteps: {self.model.num_timesteps:,} | "
                f"FPS: {fps} | Device: {self.model.device} <<<"
            )

        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train 29-DOF G1 Locomotion Policy from Scratch with RL")
    parser.add_argument('--total-timesteps', type=int, default=1000000)
    parser.add_argument('--target-vx', type=float, default=0.5)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-steps', type=int, default=2048)
    parser.add_argument('--checkpoint-interval', type=int, default=50000)
    parser.add_argument('--resume', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RL SETUP] Initializing PPO Locomotion Trainer on device: {device.upper()}")
    if device == "cuda":
        print(f"[RL SETUP] GPU: {torch.cuda.get_device_name(0)}")

    render_mode = 'none' if args.headless else 'human'
    env = G1LocomotionEnv(render_mode=render_mode, target_vx=args.target_vx)

    model_dir = base_dir / "models" / "g1_loco_rl"
    model_dir.mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"[RL SETUP] Resuming from checkpoint: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
    else:
        print("[RL SETUP] Creating brand-new 29-DOF PPO Actor-Critic policy from scratch...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
        )

    callback = LiveConsoleCallback(check_freq=1000)

    print("\n" + "=" * 65)
    print(">>> 29-DOF REINFORCEMENT LEARNING LOCOMOTION TRAINING STARTED <<<")
    if not args.headless:
        print(">>> 3D MuJoCo window launched in front of you. Watch the robot learn!")
    print(">>> Any fall, excessive tilt, or collapse triggers an instant RESET.")
    print("=" * 65 + "\n")

    trained_steps = 0
    while trained_steps < args.total_timesteps:
        chunk = min(args.checkpoint_interval, args.total_timesteps - trained_steps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=callback)
        trained_steps += chunk

        ckpt_path = model_dir / f"ppo_g1_loco_{trained_steps // 1000}k.zip"
        latest_path = model_dir / "ppo_g1_loco_latest.zip"
        model.save(ckpt_path)
        model.save(latest_path)
        print(f"\n[CHECKPOINT] Saved policy checkpoint to: {ckpt_path.name}\n")

    env.close()
    print("\n[COMPLETE] RL locomotion training finished successfully!")


if __name__ == "__main__":
    main()
