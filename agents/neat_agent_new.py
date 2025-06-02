import numpy as np
import pickle
import neat
from agents.base_agent import BaseAgent

from collections import deque

def food_accessible(state):
    """
    Returns True if there's a path from snake head to food, False otherwise.
    Uses BFS to find path while avoiding snake body.
    """
    
    snake = state['snake']
    food = state['food']
    grid_size = state['grid_size']
    
    head = snake[0]
    body_set = set(snake[1:])  # Exclude head from body
    
    # BFS setup
    queue = deque([head])
    visited = {head}
    
    # Directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    while queue:
        current = queue.popleft()
        
        # Found food
        if current == food:
            return True
        
        # Check all 4 directions
        for dx, dy in directions:
            next_pos = (current[0] + dx, current[1] + dy)
            
            # Check bounds
            if not (0 <= next_pos[0] < grid_size and 0 <= next_pos[1] < grid_size):
                continue
            
            # Check if already visited
            if next_pos in visited:
                continue
            
            # Check if it's snake body (but allow food position even if it coincides)
            if next_pos in body_set:
                continue
            
            # Add to queue and mark as visited
            queue.append(next_pos)
            visited.add(next_pos)
    
    return False

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
        Returns a 26-dimensional feature vector (floats) for the NEAT network:
        12 ray-cast normals (food/body/wall in each of Up, Right, Down, Left)
        +  4 immediate-danger flags (body or wall one cell away)
        +  4 current-direction one-hot (Up, Right, Down, Left)
        +  2 normalized dx/dy toward food
        +  2 normalized head_x/head_y
        +  1 normalized snake_length (0 to 1)
        +  1 food_accessible flag (0 or 1)
        = 26 total inputs
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
            # Default to "not found":
            dist_to_body = -1
            dist_to_wall = -1

            # 1a) Find distance to the first body segment or the wall.
            for dist in range(1, grid_size + 1):
                x = head[0] + dx * dist
                y = head[1] + dy * dist

                # If we step outside the grid, that cell is "wall":
                if not (0 <= x < grid_size and 0 <= y < grid_size):
                    dist_to_wall = dist
                    break

                # If this cell is part of the snake's body, record and stop scanning:
                if (x, y) in body:
                    dist_to_body = dist
                    break

            # If we never hit a wall in the loop (shouldn't happen), set wall = grid_size:
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

            # 1c) Normalize (1/dist) so that "closer" → larger value, or 0 if not found
            norm_food = (1.0 / dist_to_food) if (dist_to_food and dist_to_food > 0) else 0.0
            norm_body = (1.0 / dist_to_body) if (dist_to_body and dist_to_body > 0) else 0.0
            norm_wall = (1.0 / dist_to_wall)  # always > 0 because we always hit a wall at some dist

            inputs.extend([norm_food, norm_body, norm_wall])

        # 2) Immediate "danger" flags for each direction:
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

        # 6) Snake length (normalized):
        snake_length = len(state['snake'])
        snake_length_norm = snake_length / (grid_size * grid_size)  # Normalize by max possible length
        inputs.append(snake_length_norm)

        # 7) Food accessibility:
        path_exists = 1.0 if food_accessible(state) else 0.0
        inputs.append(path_exists)

        return np.array(inputs, dtype=np.float32)
        

    def select_action(self, obs, state=None):
        inp = self.extract_features(state, self.grid_size)
        output = self.net.activate(inp)
        return int(np.argmax(output))

    def seed(self, seed):
        np.random.seed(seed)
