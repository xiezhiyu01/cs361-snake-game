import os
import neat
import numpy as np
import pickle
from env.snake_game import SnakeGame
from agents.neat_agent import NEATAgent
import argparse


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# def eval_genome(genome, config):
#     from collections import defaultdict

#     game = SnakeGame(grid_size=16)
#     net = neat.nn.FeedForwardNetwork.create(genome, config)
#     state = game.reset()

#     visit_count = defaultdict(int)
#     head = state['snake'][0]
#     visit_count[tuple(head)] += 1

#     food_eaten = 0
#     max_steps = 500

#     for _ in range(max_steps):
#         features = NEATAgent.extract_features(state, grid_size=16)
#         action = np.argmax(net.activate(features))
#         state, reward, done = game.step(action)

#         head = state['snake'][0]
#         visit_count[tuple(head)] += 1

#         if reward == 1:
#             food_eaten += 1

#         if done:
#             break

#     # 1) Loop penalty: Σ (visits_to_position – 2)² for positions visited > 2 times
#     loop_penalty = sum((count - 2) ** 2 for count in visit_count.values() if count > 2)

#     # 2) Death penalty based on done_type or timeout with no food
#     dp = 0
#     dt = getattr(state, 'done_type', None) or state.get('done_type', None)
#     if dt == 'self':
#         dp = 200
#     elif dt == 'wall':
#         dp = 100
#     elif not done and food_eaten == 0:
#         dp = 300

#     # 3) Base score: (food_eaten ^ 2.5) × 100
#     base_score = (food_eaten ** 2.5) * 100 if food_eaten > 0 else 0

#     fitness = base_score - loop_penalty - dp
#     genome.fitness = fitness
#     return fitness


def eval_genome(genome, config):
    game = SnakeGame(grid_size=16)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    state = game.reset()
    
    # Get grid size from state for scalability
    grid_size = state['grid_size']
    
    food_eaten = 0
    max_steps = grid_size * grid_size * 2  # Scale with grid area
    steps_taken = 0
    
    # Track food-seeking behavior per food target
    distance_improvements = 0
    close_approaches = 0
    prev_distance = None
    
    # Define close approach distance relative to grid size
    close_distance_threshold = max(1, grid_size // 8)  # ~2 for 16x16, 1 for smaller grids
    
    for step in range(max_steps):
        # Get current position and food
        food_pos = state.get('food', None)
        
        if food_pos is not None:
            head = state['snake'][0]
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
        features = NEATAgent.extract_features(state, grid_size=grid_size)
        action = np.argmax(net.activate(features))
        state, reward, done = game.step(action)
        steps_taken += 1
        
        # Handle food consumption
        if reward == 1:
            food_eaten += 1
            prev_distance = None  # Reset for new food target
        
        if done:
            break
    
    # Simple, clear fitness calculation
    food_score = food_eaten * 1000  # Dominant reward
    
    # Bonus for moving toward food
    approach_bonus = distance_improvements * 5  # 5 points per improvement step
    
    # Bonus for getting close (cap scales with expected steps)
    max_proximity_bonus = max_steps // 10  # Scale cap with available steps
    proximity_bonus = min(close_approaches * 3, max_proximity_bonus)
    
    # Basic survival bonus (scales with grid complexity)
    max_survival_bonus = grid_size // 2  # Larger grids need more survival reward
    survival_bonus = min(steps_taken * 0.05, max_survival_bonus)
    
    fitness = food_score + approach_bonus + proximity_bonus + survival_bonus
    
    genome.fitness = fitness
    return fitness



def run_neat(config_path, n=50, path="checkpoints/best_neat_genome.pkl"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"NEAT config file not found: {config_path}")
    
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())

    winner = population.run(lambda genomes, cfg: [eval_genome(g, cfg) for g_id, g in genomes], n=n)

    with open(path, "wb") as f:
        pickle.dump(winner, f)

    print("NEAT training complete. Best genome saved.")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    # arg for model saving path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="neat_config.txt", help="Path to NEAT config file")
    parser.add_argument("--n", type=int, default=50, help="Number of generations to run")
    parser.add_argument("--path", type=str, default="checkpoints/best_neat_genome.pkl", help="Path to save the best genome")
    args = parser.parse_args()
    run_neat(args.config, args.n, args.path)
