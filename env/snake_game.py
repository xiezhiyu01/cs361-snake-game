import random
import copy

# This is for simulating a custom food respawn behavior
def simulation_generate_food(rng, state):
        while True:
            food = (
                rng.randint(0, state["grid_size"] - 1),
                rng.randint(0, state["grid_size"] - 1)
            )
            if food not in state["snake"]:
                return food

def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class SnakeGame:
    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        self.DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        self.game_rng = random.Random()
        self.simulation_rng = random.Random()
        self.reset()

    def seed(self, seed=None):
        self.game_rng.seed(seed)
        self.simulation_rng = random.Random(seed ^ 0x5DEECE66D)
        return [seed]

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)  # Initially moving right
        self.generate_food()
        self.done = False
        self.steps = 0
        return self.get_state()
    
    # def direction_from_action(self, prev_direction, action):
    #     dir_idx = self.DIRECTIONS.index(prev_direction)

    #     if action == 0:  # left
    #         dir_idx = (dir_idx - 1) % 4
    #     elif action == 2:  # right
    #         dir_idx = (dir_idx + 1) % 4
    #     # "straight" keeps dir_idx unchanged

    #     return self.DIRECTIONS[dir_idx]

    def generate_food(self):
        while True:
            food = (
                self.game_rng.randint(0, self.grid_size - 1),
                self.game_rng.randint(0, self.grid_size - 1)
            )
            if food not in self.snake:
                self.food = food
                break

    def step(self, action):
        """ Take an action in the game.
        Args:
            action (int): 0=up, 1=down, 2=left, 3=right -> Changed to 0=left, 1=straight, 2=right
        Returns:
            tuple: (state, reward, done)
        """
        
        if self.done:
            return self.get_state(), 0, True
        
        old_direction = self.direction

        self.direction = self.DIRECTIONS[action]
        new_head = (
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1]
        )

        self.steps += 1

        # Collision check and Direction check (can't go backward))
        if ((new_head in self.snake) or
            (not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size)) or
            (self.direction[0] == -old_direction[0] and self.direction[1] == -old_direction[1])):
            self.done = True
            return self.get_state(), 0, True
        

        self.snake.insert(0, new_head)
        reward = 0

        if new_head == self.food:
            reward = 1
            self.generate_food()
        else:
            self.snake.pop()

        return self.get_state(), reward, False
    
    

    def simulate_rollout_with_custom_food(self, actions, food_respawn_fn=simulation_generate_food):
        saved_rng_state = self.simulation_rng.getstate()
        total_reward = 0
        done = False
        steps_survived = 0
        improvements = 0

        for action in actions:
            reward = 0
            self.direction = self.DIRECTIONS[action]
            new_head = (
                self.snake[0][0] + self.direction[0],
                self.snake[0][1] + self.direction[1]
            )

            if (new_head in self.snake or
                not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size)):
                done = True
                break
            steps_survived += 1
            prev_distance = manhattan_distance(self.snake[0], self.food)
            new_distance = manhattan_distance(new_head, self.food)
            if new_distance < prev_distance:
                improvements += 1

            self.snake.insert(0, new_head)
            if new_head == self.food:
                reward = 1
                self.food = food_respawn_fn(self.simulation_rng, self.get_state())
            else:
                self.snake.pop()

            self.steps += 1
            total_reward += reward

        final_state = self.get_state()
        self.simulation_rng.setstate(saved_rng_state)
        return final_state, total_reward, done, steps_survived, improvements


    def get_state(self):
        return {
            'snake': self.snake.copy(),
            'food': self.food,
            'direction': self.direction,
            'grid_size': self.grid_size,
            'steps': self.steps,
            'done': self.done,
            'simulation_rng': self.simulation_rng.getstate()
        }

    # Load game + RNG state
    def load_state_for_simulation(self, state):
        self.snake = copy.deepcopy(state['snake'])
        self.food = state['food']
        self.direction = state['direction']
        self.grid_size = state['grid_size']
        self.steps = state['steps']
        self.done = state['done']
        self.simulation_rng.setstate(state['simulation_rng'])