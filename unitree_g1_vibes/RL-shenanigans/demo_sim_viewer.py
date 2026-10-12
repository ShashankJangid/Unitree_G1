import pathlib
import time
import numpy as np
from stable_baselines3 import PPO
from g1_arm_rl_env import G1ArmReachEnv

def run_visual_demo(episodes=5):
    model_path = pathlib.Path('models/ppo_g1_left_53178k.zip')
    print(f'Loading policy from {model_path}...')
    model = PPO.load(str(model_path), device='cpu')
    print('Launching MuJoCo passive viewer window...')
    env = G1ArmReachEnv(render_mode='human', right_arm=False)
    for ep in range(1, episodes + 1):
        obs, info = env.reset(seed=ep * 101)
        goal = env.p_goal
        print(f'\n[Episode {ep}/{episodes}] Reaching toward target: x={goal[0]:.2f}, y={goal[1]:.2f}, z={goal[2]:.2f}m')
        step = 0
        done = False
        while not done and step < 120:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            env.render()
            time.sleep(0.04)
            step += 1
            if terminated or truncated:
                done = True
        dist = np.linalg.norm(env.p_goal - env._fk())
        print(f'Target reached! Final hand error: {dist * 100:.1f} cm')
        for _ in range(25):
            env.render()
            time.sleep(0.04)
    print('\nVisual demo complete.')
if __name__ == '__main__':
    run_visual_demo()
