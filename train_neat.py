import os
import neat
import numpy as np
import pickle
import argparse
import logging
import sys
from env.snake_game import SnakeGame
from agents.neat_agent import NEATAgent


def setup_logger(log_path="neat_training.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, mode='w')
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    class StreamToLogger:
        def write(self, message):
            if message.strip():
                logger.info(message.strip())
        def flush(self): pass

    sys.stdout = StreamToLogger()
    sys.stderr = StreamToLogger()


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def _single_run(genome, config):
    game = SnakeGame(grid_size=16)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    state = game.reset()

    food_eaten = 0
    survival_steps = 0
    steps_since_last_food = 0
    distance_improvements = 0.0
    stalling_penalty = 0.0
    fitness = 0.0

    prev_dist = manhattan_distance(state['snake'][0], state['food'])

    for _ in range(500):
        features = NEATAgent.extract_features(state, grid_size=16)
        action = np.argmax(net.activate(features))
        state, reward, done = game.step(action)

        current_dist = manhattan_distance(state['snake'][0], state['food'])
        if current_dist < prev_dist:
            distance_improvements += 1
        else:
            distance_improvements -= 0.3
        prev_dist = current_dist

        survival_steps += 1
        steps_since_last_food += 1

        if steps_since_last_food > 100:
            stalling_penalty += 0.5

        if reward == 1:
            food_eaten += 1
            steps_since_last_food = 0

        if done:
            fitness -= 10
            break

    survival_bonus = min(survival_steps, 100) * 0.15

    fitness += (food_eaten ** 2) * 5
    fitness += survival_bonus
    fitness += distance_improvements * 0.1
    fitness -= stalling_penalty

    return fitness


def make_eval_fn(trials):
    def eval_genome(genome, config):
        fitnesses = [_single_run(genome, config) for _ in range(trials)]
        avg_fitness = np.mean(fitnesses)
        genome.fitness = avg_fitness
        return avg_fitness
    return eval_genome


def run_neat(config_path, n=50, path="checkpoints/best_neat_genome.pkl", trials=5):
    setup_logger("neat_training.log")
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

    eval_fn = make_eval_fn(trials)
    winner = population.run(lambda genomes, cfg: [eval_fn(g, cfg) for g_id, g in genomes], n=n)

    with open(path, "wb") as f:
        pickle.dump(winner, f)
    print("NEAT training complete. Best genome saved.")


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="neat_config.txt", help="Path to NEAT config file")
    parser.add_argument("--n", type=int, default=50, help="Number of generations to run")
    parser.add_argument("--path", type=str, default="checkpoints/best_neat_genome.pkl", help="Path to save the best genome")
    parser.add_argument("--trials", type=int, default=5, help="Number of eval trials per genome")
    args = parser.parse_args()
    run_neat(args.config, args.n, args.path, args.trials)
