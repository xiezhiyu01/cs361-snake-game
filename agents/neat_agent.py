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
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        head = state['snake'][0]
        body = set(state['snake'][1:])
        food = state['food']
        tail = state['snake'][-1]
        inputs = []

        # 1. Vision lines: 4 directions × (food + obstacle), values scaled by 5
        # Also write down immediate danger directions
        food_dist = []
        obstacle_dist = []
        for dir_id, (dx, dy) in enumerate(directions):
            food_d = None
            obstacle_d = None
            for dist in range(1, grid_size + 1):
                x = head[0] + dx * dist
                y = head[1] + dy * dist
                if not (0 <= x < grid_size and 0 <= y < grid_size):
                    obstacle_d = dist
                    break
                if obstacle_d is None and (x, y) in body:
                    obstacle_d = dist
                if food_d is None and (x, y) == food:
                    food_d = dist
            food_dist.append(food_d)
            obstacle_dist.append(obstacle_d)

        # Normalize distances
        inputs.extend([
            (1.0 / dist) if dist is not None else 0 for dist in food_dist
        ])
        inputs.extend([
            (1.0 / dist) if dist is not None else 0 for dist in obstacle_dist
        ])

        # Danger directions (one-hot)
        danger_dir = []
        for dist in obstacle_dist:
            if dist is None:
                danger_dir.append(0)
            elif dist == 1: # immediate danger
                danger_dir.append(5)
            elif dist <= 4: # close danger
                danger_dir.append(3)
            else: # far danger
                danger_dir.append(1)
        inputs.extend(danger_dir)

        # 2. Distance to food (normalized dx, dy)
        dx = (food[0] - head[0]) / grid_size * 5
        dy = (food[1] - head[1]) / grid_size * 5
        inputs.extend([dx, dy])

        # 2. Directon to food (one-hot)
        dx = food[0] - head[0]
        dy = food[1] - head[1]
        food_dir = [0, 0, 0, 0]  # up, down, left, right
        if abs(dx) > abs(dy):
            food_dir[0 if dx < 0 else 1] = 1
        elif dy != 0:
            food_dir[2 if dy < 0 else 3] = 1
        inputs.extend(food_dir)

        # 3. Current direction (one-hot)
        direction_map = {
            (0, 1): [1, 0, 0, 0],   # right
            (0, -1): [0, 1, 0, 0],  # left
            (1, 0): [0, 0, 1, 0],   # down
            (-1, 0): [0, 0, 0, 1],  # up
        }
        direction = direction_map.get(state['direction'], [0, 0, 0, 0])
        # inputs.extend(direction)

        # 4. Tail direction (where is the tail moving)
        if len(state['snake']) >= 2:
            tail_dx = tail[0] - state['snake'][-2][0]
            tail_dy = tail[1] - state['snake'][-2][1]
            tail_dir = direction_map.get((tail_dx, tail_dy), [0, 0, 0, 0])
        else:
            tail_dir = [0, 0, 0, 0]
        # inputs.extend(tail_dir)

        # 5. Head position (normalized)
        inputs.extend([head[0] / grid_size, head[1] / grid_size])

        return np.array(inputs, dtype=np.float32)
        

    def select_action(self, obs, state=None):
        inp = self.extract_features(state, self.grid_size)
        output = self.net.activate(inp)
        return int(np.argmax(output))

    def seed(self, seed):
        np.random.seed(seed)
