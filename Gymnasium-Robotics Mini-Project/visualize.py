from stable_baselines3 import PPO
from envs.reach_env import ReachEnv
import time

env = ReachEnv(grid_size=10)
model = PPO.load("ppo_reach")

obs, _ = env.reset()
for step in range(30):
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    print(f"Step {step}: Action={action}, Reward={reward}")
    env.render()
    time.sleep(0.2)
    if done:
        print("🎯 Target reached!")
        break
