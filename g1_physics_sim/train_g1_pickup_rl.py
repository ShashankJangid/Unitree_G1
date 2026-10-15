from __future__ import annotations
import argparse
import math
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

from g1_pickup_rl_env import G1PickupEnv


class PickupTrainingCallback(BaseCallback):
    def __init__(self, check_freq: int = 50):
        super().__init__()
        self.check_freq = check_freq
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "dist" in info and self.locals.get("dones", [False])[0]:
                self.episode_count += 1
                dist = info.get("dist", 0.0)
                wp = info.get("waist_pitch", 0.0)
                crouch = info.get("crouch", 0.0)
                step = info.get("step", 0)
                tz = info.get("target_z", 0.0)
                close_str = "[GRASP REACHED!]" if dist < 0.05 else "[TIMED OUT]"

                print(
                    f"Episode {self.episode_count:4d} | Steps: {step:3d} | "
                    f"Dist: {dist*100:4.1f}cm | WaistBend: {math.degrees(wp):4.1f}deg | "
                    f"Crouch: {crouch:4.2f} | ObjHeight: {tz:4.2f}m | {close_str}"
                )

        if self.n_calls % self.check_freq == 0:
            elapsed = max(1e-3, (time.time() - self.model.start_time))
            fps = int(self.model.num_timesteps / elapsed)
            print(
                f">>> Timesteps: {self.model.num_timesteps:,} | "
                f"FPS: {fps} | Device: {self.model.device} <<<"
            )

        return True


def parse_args():
    parser = argparse.ArgumentParser(description="Train G1 Humanoid Bending Down & Object Pickup with RL")
    parser.add_argument("--total-timesteps", type=int, default=500000)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--checkpoint-interval", type=int, default=25000)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[PICKUP RL] Initializing G1 Bending & Pickup RL Trainer on {device.upper()}...")
    if device == "cuda":
        print(f"[PICKUP RL] GPU: {torch.cuda.get_device_name(0)}")

    render_mode = "none" if args.headless else "human"
    env = G1PickupEnv(render_mode=render_mode)

    model_dir = base_dir / "models" / "g1_pickup_rl"
    model_dir.mkdir(parents=True, exist_ok=True)

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"[PICKUP RL] Resuming from checkpoint: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
    else:
        print("[PICKUP RL] Creating new whole-body PPO policy for bending & pickup...")
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

    callback = PickupTrainingCallback(check_freq=50)

    print("\n" + "=" * 80)
    print(">>> G1 WHOLE-BODY BENDING DOWN & OBJECT PICKUP RL TRAINING STARTED <<<")
    print("=" * 80)
    print(f"  Target Timesteps     : {args.total_timesteps:,}")
    print(f"  Live 3D Rendering    : {not args.headless}")
    print(f"  Controlled Joints    : Waist (Pitch/Roll/Yaw) + Knees Crouch + 7-DOF Arm + Hand Grasp")
    print(f"  Checkpoints Location : {model_dir}")
    print("=" * 80 + "\n")

    steps_completed = 0
    while steps_completed < args.total_timesteps:
        chunk = min(args.checkpoint_interval, args.total_timesteps - steps_completed)
        model.learn(
            total_timesteps=chunk,
            reset_num_timesteps=False,
            callback=callback
        )
        steps_completed += chunk

        ckpt_path = model_dir / f"g1_pickup_ppo_{steps_completed // 1000}k.zip"
        model.save(str(ckpt_path))
        print(f"[CHECKPOINT] Saved policy checkpoint to: {ckpt_path.name}")

    final_model_path = model_dir / "g1_pickup_ppo_final.zip"
    model.save(str(final_model_path))
    print(f"[TRAINING COMPLETE] Final policy saved to: {final_model_path}")
    env.close()


if __name__ == "__main__":
    main()
