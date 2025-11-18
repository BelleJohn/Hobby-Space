import gymnasium as gym
from stable_baselines3 import PPO
from envs.reach_env import ReachEnv
# build custom environment
env = ReachEnv(grid_size=10)

# build PPO model
model = PPO("MlpPolicy", env, verbose=1)

# train for 50,000 steps
model.learn(total_timesteps=50000)

# save model
model.save("ppo_reach")
