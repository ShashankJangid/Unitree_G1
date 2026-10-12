import pathlib
import numpy as np
from stable_baselines3 import PPO
from g1_arm_rl_env import G1ArmReachEnv

def test_policy(num_episodes=5):
    model_path = pathlib.Path('models/ppo_g1_left_53178k.zip')
    print(f'Loading policy from: {model_path}...')
    model = PPO.load(str(model_path), device='cpu')
    print('Policy successfully loaded.')
    env = G1ArmReachEnv(render_mode='none', right_arm=False)
    success_count = 0
    all_final_distances = []
    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(seed=ep * 42)
        goal = env.p_goal.copy()
        print(f'\n--- Episode {ep} ---')
        print(f'Target Goal Position (x, y, z): {goal[0]:.3f}, {goal[1]:.3f}, {goal[2]:.3f} m')
        init_dist = np.linalg.norm(goal - env._fk())
        print(f'Initial hand-to-goal distance: {init_dist * 100:.1f} cm')
        done = False
        step = 0
        min_dist = init_dist
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action)
            dist = step_info['dist']
            min_dist = min(min_dist, dist)
            step += 1
            if terminated or truncated or step >= env.max_steps:
                done = True
        final_dist = np.linalg.norm(goal - env._fk())
        all_final_distances.append(final_dist)
        is_success = final_dist < 0.05
        if is_success:
            success_count += 1
        print(f'Finished in {step} steps.')
        print(f'Final distance: {final_dist * 100:.1f} cm (Closest: {min_dist * 100:.1f} cm)')
        print(f"Status: {('[SUCCESS]' if is_success else '[FAIL]')}")
    print('\n================ EVALUATION SUMMARY ================')
    print(f'Episodes tested: {num_episodes}')
    print(f'Success rate (dist < 5cm): {success_count}/{num_episodes} ({success_count / num_episodes * 100:.1f}%)')
    print(f'Mean final distance: {np.mean(all_final_distances) * 100:.1f} cm')
if __name__ == '__main__':
    test_policy()
