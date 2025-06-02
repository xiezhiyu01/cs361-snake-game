from agents.base_agent import BaseAgent
import numpy as np
from env.snake_game import SnakeGame


def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class GAAgent(BaseAgent):
    def __init__(self, grid_size=16, population_size=50, rollout_length=15, elite_frac=0.2, mutation_rate=0.1, num_generations=3):
        super().__init__(grid_size)
        self.population_size = population_size
        self.rollout_length = rollout_length
        self.elite_frac = elite_frac
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations

    def _fitness_function(self, rollout_results, grid_size):
        """
        Updated fitness function matching the NEAT agent's approach
        rollout_results should contain: total_reward, steps_survived, distance_improvements, close_approaches
        """
        total_reward, steps_survived, distance_improvements, close_approaches = rollout_results
        
        # Simple, clear fitness calculation (matching NEAT agent)
        food_score = total_reward * 1000  # Dominant reward (total_reward = food_eaten)
        
        # Bonus for moving toward food
        approach_bonus = distance_improvements * 5  # 5 points per improvement step
        
        # Bonus for getting close (cap scales with expected steps)
        max_steps = grid_size * grid_size * 2  # Scale with grid area
        max_proximity_bonus = max_steps // 10  # Scale cap with available steps
        proximity_bonus = min(close_approaches * 3, max_proximity_bonus)
        
        # Basic survival bonus (scales with grid complexity)
        max_survival_bonus = grid_size // 2  # Larger grids need more survival reward
        survival_bonus = min(steps_survived * 0.05, max_survival_bonus)
        
        fitness = food_score + approach_bonus + proximity_bonus + survival_bonus
        return fitness

    def _crossover(self, p1, p2):
        return [p1[i] if np.random.rand() < 0.5 else p2[i] for i in range(self.rollout_length)]

    def _mutate(self, sequence):
        return [
            np.random.choice(4) if np.random.rand() < self.mutation_rate else action
            for action in sequence
        ]
    
    def _initialize_rollout_with_initial_action(self, initial_action):
        """to prevent snake from going back on itself, we need to know previous action"""
        action_sequence = []
        current_action = initial_action
        while len(action_sequence) < self.rollout_length:
            action = np.random.choice(4)
            # avoid snake to go back on itself
            # actions are mapped from 0 1 2 3 to up right down left
            # if abs(current_action - action) == 2:
            #     continue
            action_sequence.append(action)
            current_action = action
        return action_sequence

    def _simulate_rollout_with_fitness_tracking(self, sim, action_sequence, grid_size):
        """
        Simulate rollout while tracking distance improvements and close approaches
        Returns: (total_reward, steps_survived, distance_improvements, close_approaches)
        """
        total_reward = 0
        steps_survived = 0
        distance_improvements = 0
        close_approaches = 0
        prev_distance = None
        
        # Define close approach distance relative to grid size
        close_distance_threshold = max(1, grid_size // 8)  # ~2 for 16x16, 1 for smaller grids
        
        for action in action_sequence:
            # Get current state before action
            current_state = sim.get_state()
            food_pos = current_state.get('food', None)
            
            if food_pos is not None:
                head = current_state['snake'][0]
                current_distance = manhattan_distance(head, food_pos)
                
                # Count close approaches (scaled to grid size)
                if current_distance <= close_distance_threshold:
                    close_approaches += 1
                
                # Track improvement (only if we have previous distance to same food)
                if prev_distance is not None:
                    if current_distance < prev_distance:
                        distance_improvements += 1
                
                # Update for next comparison
                prev_distance = current_distance
            
            # Take action
            _, reward, done = sim.step(action)
            total_reward += reward
            steps_survived += 1
            
            # Handle food consumption
            if reward == 1:
                prev_distance = None  # Reset for new food target
            
            if done:
                break
        
        return total_reward, steps_survived, distance_improvements, close_approaches

    def select_action(self, obs, state):
        best_sequence = None
        best_score = float('-inf')

        # Get current direction
        current_direction = state['direction']
        direction_to_action = {
            (-1, 0): 0,  # up
            (0, 1): 1,   # right
            (1, 0): 2,   # down
            (0, -1): 3   # left
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
                
                # Use enhanced simulation with fitness tracking
                rollout_results = self._simulate_rollout_with_fitness_tracking(sim, seq, self.grid_size)
                fitness = self._fitness_function(rollout_results, self.grid_size)
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

            population = new_population

        return best_sequence[0] if best_sequence else np.random.choice(4)