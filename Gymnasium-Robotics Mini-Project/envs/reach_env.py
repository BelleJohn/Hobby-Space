import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ReachEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(ReachEnv, self).__init__()
        
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        # Observation: [agent_x, agent_y, target_x, target_y]
        self.observation_space = spaces.Box(
            low=0, high=grid_size, shape=(4,), dtype=np.float32
        )

        self.agent_pos = None
        self.target_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.random.randint(0, self.grid_size, size=2)
        self.target_pos = np.random.randint(0, self.grid_size, size=2)
        obs = np.concatenate((self.agent_pos, self.target_pos)).astype(np.float32)
        return obs, {}

    def step(self, action):
        # Move agent
        if action == 0:   # up
            self.agent_pos[1] += 1
        elif action == 1: # down
            self.agent_pos[1] -= 1
        elif action == 2: # left
            self.agent_pos[0] -= 1
        elif action == 3: # right
            self.agent_pos[0] += 1

        # Keep inside bounds
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Compute reward
        distance = np.linalg.norm(self.agent_pos - self.target_pos)
        done = distance < 1.0
        reward = 10.0 if done else -distance * 0.1

        obs = np.concatenate((self.agent_pos, self.target_pos)).astype(np.float32)
        return obs, reward, done, False, {}

    def render(self):
        grid = np.zeros((self.grid_size + 1, self.grid_size + 1))
        grid[self.agent_pos[1], self.agent_pos[0]] = 1
        grid[self.target_pos[1], self.target_pos[0]] = 0.5
        print(grid)
