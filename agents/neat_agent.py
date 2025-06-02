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
        """
        Returns a 25-dimensional feature vector (floats) for the NEAT network:
           12 ray-cast normals (food/body/wall in each of Up, Right, Down, Left)
         +  4 immediate-danger flags (body or wall one cell away)
         +  4 current-direction one-hot (Up, Right, Down, Left)
         +  2 normalized dx/dy toward food
         +  2 normalized head_x/head_y
         +  1 normalized steps_left (0 to 1)
         = 25 total inputs
        """

        head = state['snake'][0]
        body = set(state['snake'][1:])
        food = state['food']
        curr_dir = state['direction']   # e.g. (0,1) for right, (-1,0) for up, etc.

        inputs = []

        # 1) Ray-casting in four cardinal directions: Up, Right, Down, Left
        #    For each direction we will compute:
        #      dist_to_food (if on that ray), dist_to_body, dist_to_wall.
        #    Then normalize them via (1.0/d) or 0 if not found.
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        dist_to_body_list = []
        dist_to_wall_list = []
        dist_to_food_list = []

        for (dx, dy) in directions:
            # Default to “not found”:
            dist_to_body = -1
            dist_to_wall = -1

            # 1a) Find distance to the first body segment or the wall.
            for dist in range(1, grid_size + 1):
                x = head[0] + dx * dist
                y = head[1] + dy * dist

                # If we step outside the grid, that cell is “wall”:
                if not (0 <= x < grid_size and 0 <= y < grid_size):
                    dist_to_wall = dist
                    break

                # If this cell is part of the snake's body, record and stop scanning:
                if (x, y) in body:
                    dist_to_body = dist
                    break

            # If we never hit a wall in the loop (shouldn’t happen), set wall = grid_size:
            if dist_to_wall < 0:
                dist_to_wall = grid_size

            # 1b) Scan again (up to dist_to_wall) for whether the food is on that ray:
            dist_to_food = -1
            for dist in range(1, dist_to_wall):
                x = head[0] + dx * dist
                y = head[1] + dy * dist
                if (x, y) == food:
                    dist_to_food = dist
                    break

            dist_to_body_list.append(dist_to_body)
            dist_to_wall_list.append(dist_to_wall)
            dist_to_food_list.append(dist_to_food)

            # 1c) Normalize (1/dist) so that “closer” → larger value, or 0 if not found
            norm_food = (1.0 / dist_to_food) if (dist_to_food and dist_to_food > 0) else 0.0
            norm_body = (1.0 / dist_to_body) if (dist_to_body and dist_to_body > 0) else 0.0
            norm_wall = (1.0 / dist_to_wall)  # always > 0 because we always hit a wall at some dist

            inputs.extend([norm_food, norm_body, norm_wall])

        # 2) Immediate “danger” flags for each direction:
        #    If dist_to_body == 1 or dist_to_wall == 1, mark that dir as dangerous (1.0), else 0.0.
        danger_dir = [0.0, 0.0, 0.0, 0.0]
        for i in range(4):
            if dist_to_body_list[i] == 1 or dist_to_wall_list[i] == 1:
                danger_dir[i] = 1.0
        inputs.extend(danger_dir)

        # 3) Current direction one-hot (Up, Right, Down, Left)
        direction_map = {
            (-1, 0): [1.0, 0.0, 0.0, 0.0],  # Up
            (0, 1):  [0.0, 1.0, 0.0, 0.0],  # Right
            (1, 0):  [0.0, 0.0, 1.0, 0.0],  # Down
            (0, -1): [0.0, 0.0, 0.0, 1.0],  # Left
        }
        curr_dir_onehot = direction_map.get(curr_dir, [0.0, 0.0, 0.0, 0.0])
        inputs.extend(curr_dir_onehot)

        # 4) Normalized (dx, dy) to food:
        dx_norm = (food[0] - head[0]) / float(grid_size)
        dy_norm = (food[1] - head[1]) / float(grid_size)
        inputs.extend([dx_norm, dy_norm])

        # 5) Normalized head position
        inputs.extend([head[0] / float(grid_size), head[1] / float(grid_size)])

        # Final feature vector length = 12 (ray) + 4 (danger) + 4 (curr dir) + 2 (dx,dy) + 2 (head pos)
        #                        = 24 features
        # If you want exactly 20 inputs, you can drop the “danger” flags (4 dims) and keep everything else,
        # for a total of 20. Below, we’ll assume you want all 24. If you do drop danger_dir, remember to change num_inputs.
        # 6) Steps left (normalized):
        #    If you want to explicitly feed steps_left, include it here:
        steps_left = (grid_size * grid_size) - state['steps_since_food']
        steps_left_norm = max(0,steps_left) / (grid_size * grid_size)
        inputs.append(steps_left_norm)

        return np.array(inputs, dtype=np.float32)
        

    def select_action(self, obs, state=None):
        inp = self.extract_features(state, self.grid_size)
        output = self.net.activate(inp)
        return int(np.argmax(output))

    def seed(self, seed):
        np.random.seed(seed)
