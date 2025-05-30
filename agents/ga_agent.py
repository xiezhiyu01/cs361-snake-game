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

    def select_action(self, obs, state):
        best_sequence = None
        best_score = float('-inf')

        # Initialize population
        population = [
            [np.random.choice(4) for _ in range(self.rollout_length)]
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

