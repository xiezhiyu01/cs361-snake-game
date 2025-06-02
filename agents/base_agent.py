import numpy as np

class BaseAgent:
    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        self.directions = {
            0: (-1, 0),  # up
            1: (0, 1), # right
            2: (1, 0),   # down
            3: (0, -1)  # left
        }

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def select_action(self, obs, state):
        raise NotImplementedError("select_action() must be implemented by subclass.")
