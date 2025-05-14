import numpy as np
import pickle
import neat
from agents.base_agent import BaseAgent

class NEATAgent(BaseAgent):
    def __init__(self, path="checkpoints/best_neat_genome.pkl", config_path="neat_config.txt", grid_size=16):
        super().__init__(grid_size)
        self.config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path
        )
        with open(path, "rb") as f:
            genome = pickle.load(f)
        self.net = neat.nn.FeedForwardNetwork.create(genome, self.config)

    def preprocess(self, state):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for x, y in state['snake']:
            grid[x, y] = -0.5
        fx, fy = state['food']
        grid[fx, fy] = 1.0
        return grid.flatten()

    def select_action(self, obs, state=None):
        inp = self.preprocess(state)
        output = self.net.activate(inp)
        return int(np.argmax(output))

    def seed(self, seed):
        np.random.seed(seed)
