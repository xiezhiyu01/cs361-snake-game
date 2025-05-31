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

    @staticmethod
    def extract_features(state, grid_size=16):
        directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                    (1, 0), (1, -1), (0, -1), (-1, -1)]
        head = state['snake'][0]
        snake_body = set(state['snake'][1:])
        food = state['food']
        inputs = []

        # 1. Vision lines: 8 directions × 3 distances
        for dx, dy in directions:
            food_dist = 0
            body_dist = 0
            wall_dist = 0
            for dist in range(1, grid_size + 1):
                x = head[0] + dx * dist
                y = head[1] + dy * dist
                if not (0 <= x < grid_size and 0 <= y < grid_size):
                    wall_dist = 1.0 / dist
                    break
                if (food_dist == 0) and (x, y) == food:
                    food_dist = 1.0 / dist
                if (body_dist == 0) and ((x, y) in snake_body):
                    body_dist = 1.0 / dist
            if wall_dist == 0:
                wall_dist = 1.0 / grid_size
            inputs.extend([food_dist, body_dist, wall_dist])

        # 2. Snake direction one-hot: 4 dims
        direction_map = {
            (0, 1): [1, 0, 0, 0],   # right
            (0, -1): [0, 1, 0, 0],  # left
            (1, 0): [0, 0, 1, 0],   # down
            (-1, 0): [0, 0, 0, 1]   # up
        }
        direction = direction_map.get(state['direction'], [0, 0, 0, 0])
        inputs.extend(direction)

        # 3. Food direction one-hot: 4 dims
        food_dx = food[0] - head[0]
        food_dy = food[1] - head[1]
        food_dir = [
            int(food_dx > 0), int(food_dx < 0),  # down, up
            int(food_dy > 0), int(food_dy < 0)   # right, left
        ]
        inputs.extend(food_dir)

        # 4. Food vector (dx, dy), normalized
        inputs.append(food_dx / grid_size)
        inputs.append(food_dy / grid_size)

        # 5. Manhattan distance to food (normalized)
        manhattan_dist = abs(food_dx) + abs(food_dy)
        inputs.append(manhattan_dist / (2 * grid_size))

        # 6. Snake length (raw)
        inputs.append(len(state['snake']))

        return np.array(inputs, dtype=np.float32)


    def select_action(self, obs, state=None):
        inp = self.extract_features(state, self.grid_size)
        output = self.net.activate(inp)
        return int(np.argmax(output))

    def seed(self, seed):
        np.random.seed(seed)
