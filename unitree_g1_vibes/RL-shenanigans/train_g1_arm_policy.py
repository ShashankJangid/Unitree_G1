from __future__ import annotations
import argparse
import datetime as _dt
import pathlib
import pprint
from typing import List
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
from g1_arm_rl_env import make_env
ALGOS = {'ppo': PPO, 'sac': SAC, 'td3': TD3}

def parse_args():
    ap = argparse.ArgumentParser(description='Train RL policy for G-1 arm reach task (MuJoCo)')
    ap.add_argument('--algo', choices=ALGOS.keys(), default='ppo', help='RL algorithm (default: %(default)s)')
    ap.add_argument('--total-steps', type=int, default=50000000, help='Total environment steps (default: %(default)s)')
    ap.add_argument('--num-envs', type=int, default=16, help='Parallel envs (default: %(default)s)')
    ap.add_argument('--right-arm', action='store_true', help='Train right arm instead of left')
    ap.add_argument('--checkpoint', type=int, default=250000, help='Checkpoint interval (steps)')
    ap.add_argument('--headless', action='store_true', help='Disable MuJoCo viewer even if DISPLAY present')
    ap.add_argument('--show-envs', type=int, default=1, help='How many parallel env viewers to launch (default: %(default)s)')
    ap.add_argument('--resume-from', type=pathlib.Path, metavar='CKPT', help='Path to previously saved *.zip checkpoint to continue training from')
    return ap.parse_args()

def make_vec(n: int, right_arm: bool, headless: bool, show_envs: int) -> DummyVecEnv:
    env_fns: List[callable] = []
    for idx in range(n):

        def _thunk(idx=idx):
            rm = 'human' if idx < show_envs and (not headless) else 'none'
            return make_env(right_arm=right_arm, render_mode=rm)
        env_fns.append(_thunk)
    return DummyVecEnv(env_fns)

def main() -> None:
    args = parse_args()
    run_name = _dt.datetime.now().strftime('%Y%m%d-%H%M%S')
    run_name += '-rArm' if args.right_arm else '-lArm'
    log_dir = pathlib.Path('runs/g1_arm_reach') / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    vec_env = make_vec(args.num_envs, args.right_arm, args.headless, args.show_envs)
    model_cls = ALGOS[args.algo]
    if args.resume_from is not None:
        if not args.resume_from.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {args.resume_from}')
        print(f"Resuming training from '{args.resume_from}' …")
        model = model_cls.load(args.resume_from, env=vec_env, device='auto')
        start_steps = int(model.num_timesteps)
    else:
        model = model_cls('MlpPolicy', vec_env, verbose=1, tensorboard_log=str(log_dir), device='auto')
        start_steps = 0
    print('Training config:')
    pprint.pprint(vars(args))
    checkpoint_dir = pathlib.Path('models')
    checkpoint_dir.mkdir(exist_ok=True)
    steps = start_steps
    target_steps = args.total_steps + start_steps
    while steps < target_steps:
        next_chunk = min(args.checkpoint, target_steps - steps)
        model.learn(total_timesteps=next_chunk, reset_num_timesteps=False, tb_log_name='RL')
        steps += next_chunk
        ckpt_path = checkpoint_dir / f"{args.algo}_g1_{('right' if args.right_arm else 'left')}_{steps // 1000}k.zip"
        model.save(ckpt_path)
        try:
            rel = ckpt_path.relative_to(pathlib.Path.cwd())
            print('Checkpoint saved →', rel)
        except ValueError:
            print('Checkpoint saved →', ckpt_path)
    final_path = checkpoint_dir / f"{args.algo}_g1_{('right' if args.right_arm else 'left')}_final.zip"
    model.save(final_path)
    print('Final policy saved to', final_path)
if __name__ == '__main__':
    main()
