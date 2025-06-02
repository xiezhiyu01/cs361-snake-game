from agents.base_agent import BaseAgent
import numpy as np
import random

class GreedyAgent(BaseAgent):
    def __init__(self, grid_size=16):
        super().__init__(grid_size)

    def select_action(self, obs, state):
        head = state['snake'][0]
        food = state['food']
        body = set(state['snake'][1:])

        def is_safe(pos):
            x, y = pos
            return (0 <= x < self.grid_size and 0 <= y < self.grid_size and pos not in body)

        candidates = []
        for action, delta in self.directions.items():
            new_pos = (head[0] + delta[0], head[1] + delta[1])
            if is_safe(new_pos):
                dist = abs(new_pos[0] - food[0]) + abs(new_pos[1] - food[1])
                candidates.append((dist, action))

        if candidates:
            # Sort candidates by distance to food, then randomly select one
            candidates.sort(key=lambda x: x[0])
            min_distance = candidates[0][0]
            best_candidates = [action for dist, action in candidates if dist == min_distance]
            return random.choice(best_candidates)
        else:
            return random.choice(list(self.directions.keys()))
