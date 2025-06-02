import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from env.snake_game import SnakeGame
from env.renderer import SnakeRenderer

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=16, agent_name=None, render_save_dir=None):
        super(SnakeEnv, self).__init__()
        self.grid_size = grid_size
        # import pdb
        # pdb.set_trace()
        self.agent_name = agent_name
        self.game = SnakeGame(grid_size)
        self.renderer = SnakeRenderer(grid_size, save_dir=render_save_dir)

        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.grid_size, self.grid_size, 3),
            dtype=np.uint8
        )

    def seed(self, seed=None):
        np.random.seed(seed)
        self.game.seed(seed)
        return [seed]

    def reset(self):
        state = self.game.reset()
        return self._get_obs(state)

    def step(self, action):
        state, reward, done = self.game.step(action)
        obs = self._get_obs(state)
        info = {
            'steps': state['steps'],
            'reward': reward
        }
        adjusted_reward = reward
        # Adjust the reward based on the agent type
        if self.agent_name == 'PPO':
            if reward == 1:
                adjusted_reward = 1
            else:
                adjusted_reward = -0.005
                dist = abs(state['snake'][0][0] - state['food'][0]) + abs(state['snake'][0][1] - state['food'][1])
                adjusted_reward += 0.01 * (1 - dist / (self.grid_size * 2))
        return obs, adjusted_reward, done, info

    def get_state(self):
        return self.game.get_state()

    def _get_obs(self, state):
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        grid[state['food'][0], state['food'][1], 0] = 255 # Food
        for idx, pos in enumerate(state['snake']):
            if idx == 0:
                grid[pos[0], pos[1], 1] = 255 # Head
            else:
                grid[pos[0], pos[1], 2] = 255
        return grid

    def render(self, mode='human'):
        state = self.game.get_state()
        self.renderer.render(state)
