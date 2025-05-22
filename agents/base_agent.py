import numpy as np

class BaseAgent:
    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        # self.directions = {
        #     0: (-1, 0),  # up
        #     1: (1, 0),   # down
        #     2: (0, -1),  # left
        #     3: (0, 1)    # right
        # }
        self.directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Clockwise: Up, Right, Down, Left

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
    
    def direction_from_action(self, prev_direction, action):
        dir_idx = self.directions.index(prev_direction)

        if action == 0:  # left
            dir_idx = (dir_idx - 1) % 4
        elif action == 2:  # right
            dir_idx = (dir_idx + 1) % 4
        # "straight" keeps dir_idx unchanged

        return self.directions[dir_idx]

    def select_action(self, obs, state):
        raise NotImplementedError("select_action() must be implemented by subclass.")
