from agents.base_agent import BaseAgent
import random

class RandomAgent(BaseAgent):
    def __init__(self, grid_size=16):
        super().__init__(grid_size)

    def select_action(self, obs, state):
        """ Select a random action."""
        return random.randint(0, 3)