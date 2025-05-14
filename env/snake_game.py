import random

class SnakeGame:
    def __init__(self, grid_size=16):
        self.grid_size = grid_size
        self.DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.reset()

    def seed(self, seed=None):
        random.seed(seed)
        return [seed]

    def reset(self):
        self.snake = [(self.grid_size // 2, self.grid_size // 2)]
        self.direction = (0, 1)  # Initially moving right
        self.generate_food()
        self.done = False
        self.steps = 0
        return self.get_state()

    def generate_food(self):
        while True:
            food = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            if food not in self.snake:
                self.food = food
                break

    def step(self, action):
        """ Take an action in the game.
        Args:
            action (int): 0=up, 1=down, 2=left, 3=right
        Returns:
            tuple: (state, reward, done)
        """
        
        if self.done:
            return self.get_state(), 0, True

        self.direction = self.DIRECTIONS[action]
        new_head = (
            self.snake[0][0] + self.direction[0],
            self.snake[0][1] + self.direction[1]
        )

        self.steps += 1

        # Collision check
        if (new_head in self.snake or
            not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size)):
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

    def get_state(self):
        return {
            'snake': self.snake.copy(),
            'food': self.food,
            'direction': self.direction,
            'grid_size': self.grid_size,
            'steps': self.steps,
            'done': self.done
        }
