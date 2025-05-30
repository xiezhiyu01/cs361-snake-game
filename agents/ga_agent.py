from agents.base_agent import BaseAgent
import numpy as np
from env.snake_game import SnakeGame


class GAAgent(BaseAgent):
    def __init__(self, grid_size=16, population_size=50, rollout_length=15, elite_frac=0.2, mutation_rate=0.1, num_generations=3):
        super().__init__(grid_size)
        self.population_size = population_size
        self.rollout_length = rollout_length
        self.elite_frac = elite_frac
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations

    def _fitness_function(self, total_reward, steps_survived, improvements):
        return 100.0 * total_reward + (steps_survived * 0.5) + improvements * 0.1

    def _crossover(self, p1, p2):
        return [p1[i] if np.random.rand() < 0.5 else p2[i] for i in range(self.rollout_length)]

    def _mutate(self, sequence):
        return [
            np.random.choice(list(range(3))) if np.random.rand() < self.mutation_rate else action
            for action in sequence
        ]
    
    def _initialize_rollout_with_initial_action(self, initial_action):
        """to prevent snake from going back on itself, we need to know previous action"""
        action_sequence = []
        current_action = initial_action
        while len(action_sequence) < self.rollout_length:
            action = np.random.choice(4)
            # avoid snake to go back on itself
            # actions are mapped from 0 1 2 3 to up down left right
            if action == 0:
                if current_action == 1:
                    continue
            if action == 1:
                if current_action == 0:
                    continue
            if action == 2:
                if current_action == 3:
                    continue
            if action == 3:
                if current_action == 2:
                    continue
            action_sequence.append(action)
            current_action = action
        return action_sequence

    def select_action(self, obs, state):
        best_sequence = None
        best_score = float('-inf')

        # remember initial direction is right (action 3)
        current_direction = state['direction']
        direction_to_action = {
            (-1, 0):0,  # up
            (1, 0):1,   # down
            (0, -1):2,  # left
            (0, 1):3    # right
        }
        initial_action = direction_to_action[current_direction]
        population = [
            self._initialize_rollout_with_initial_action(initial_action)
            for _ in range(self.population_size)
        ]

        for gen in range(self.num_generations):
            fitness_scores = []

            for seq in population:
                sim = SnakeGame(grid_size=self.grid_size)
                sim.load_state_for_simulation(state)
                _, total_reward, done, steps_survived, improvements = sim.simulate_rollout_with_custom_food(seq)
                fitness = self._fitness_function(total_reward, steps_survived, improvements)
                fitness_scores.append((fitness, seq))

            fitness_scores.sort(reverse=True, key=lambda x: x[0])
            num_elites = max(1, int(self.elite_frac * self.population_size))
            elites = [seq for (_, seq) in fitness_scores[:num_elites]]

            if fitness_scores[0][0] > best_score:
                best_score = fitness_scores[0][0]
                best_sequence = fitness_scores[0][1]

            # Reproduce next generation
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent_indices = np.random.choice(len(elites), size=2, replace=False)
                parent1, parent2 = elites[parent_indices[0]], elites[parent_indices[1]]
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population  # evolve to next gen

        return best_sequence[0] if best_sequence else np.random.choice(3)

