import os
import neat
import numpy as np
import pickle
from env.snake_game import SnakeGame
from agents.neat_agent import NEATAgent
import argparse


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def eval_genome(genome, config):
    game = SnakeGame(grid_size=16)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    state = game.reset()

    food_eaten = 0
    survival_steps = 0
    steps_since_last_food = 0
    distance_improvements = []

    prev_dist = manhattan_distance(state['snake'][0], state['food'])

    for _ in range(500):  # Max survival_steps
        features = NEATAgent.extract_features(state, grid_size=16)
        action = np.argmax(net.activate(features))
        state, reward, done = game.step(action)

        current_dist = manhattan_distance(state['snake'][0], state['food'])
        distance_improvements.append(prev_dist - current_dist)
        prev_dist = current_dist

        survival_steps += 1
        steps_since_last_food += 1

        if reward == 1:
            food_eaten += 1
            steps_since_last_food = 0  # reset on eating

        if done:
            break

    # Final fitness
    fitness = (
        food_eaten * 10
        + survival_steps * 0.1
        + sum(distance_improvements) * 0.05
        - (steps_since_last_food if steps_since_last_food > 50 else 0)  # penalize stalling
    )
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
