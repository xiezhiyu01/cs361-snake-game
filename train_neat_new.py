import os
import neat
import numpy as np
import pickle
import glob
from env.snake_game import SnakeGame
from agents.neat_agent import NEATAgent
import argparse
import multiprocessing as mp
from functools import partial
import sys
from datetime import datetime


class LoggingReporter(neat.reporting.BaseReporter):
    def __init__(self, log_file):
        self.log_file = log_file
    
    def start_generation(self, generation):
        with open(self.log_file, 'a') as f:
            f.write(f"\n=== Generation {generation} ===\n")
            f.flush()
    
    def post_evaluate(self, config, population, species, best_genome):
        best_fitness = best_genome.fitness if best_genome.fitness is not None else 0
        avg_fitness = sum(g.fitness for g in population.values() if g.fitness is not None) / len(population)
        
        with open(self.log_file, 'a') as f:
            f.write(f"Best fitness: {best_fitness:.2f}, Average fitness: {avg_fitness:.2f}\n")
            f.write(f"Population size: {len(population)}, Species: {len(species.species)}\n")
            f.flush()


class BestGenomeSaver(neat.reporting.BaseReporter):
    def __init__(self, save_path, log_file=None):
        self.save_path = save_path
        self.best_fitness = float('-inf')
        self.log_file = log_file
    
    def post_evaluate(self, config, population, species, best_genome):
        current_fitness = best_genome.fitness if best_genome.fitness is not None else float('-inf')
        
        if current_fitness > self.best_fitness:
            self.best_fitness = current_fitness
            with open(self.save_path, "wb") as f:
                pickle.dump(best_genome, f)
            
            message = f"New best genome saved! Fitness: {current_fitness:.2f}"
            print(message)
            
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(f"{message}\n")
                    f.flush()

# just added
class LatestGenomeSaver(neat.reporting.BaseReporter):
    def __init__(self, experiment_tag, log_file=None):
        self.experiment_tag = experiment_tag
        self.log_file = log_file
    
    def post_evaluate(self, config, population, species, best_genome):
        # Save the best genome from current generation
        latest_path = f"checkpoints/{self.experiment_tag}-latest.pkl"
        with open(latest_path, "wb") as f:
            pickle.dump(best_genome, f)
        
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"Latest genome saved: {best_genome.fitness:.2f}\n")
                f.flush()

class CleaningCheckpointer(neat.Checkpointer):
    """Checkpointer that automatically cleans up old checkpoints"""
    def __init__(self, generation_interval=1, time_interval_seconds=300, 
                 filename_prefix='neat-checkpoint-', experiment_tag="default"):
        super().__init__(generation_interval, time_interval_seconds, filename_prefix)
        self.experiment_tag = experiment_tag
    
    def save_checkpoint(self, config, population, species_set, generation):
        # Save the new checkpoint
        super().save_checkpoint(config, population, species_set, generation)
        
        # Clean up old checkpoints immediately after saving
        self.clean_old_checkpoints()
    
    def clean_old_checkpoints(self):
        """Keep only the latest checkpoint"""
        pattern = f"checkpoints/{self.experiment_tag}-checkpoint*"
        checkpoints = glob.glob(pattern)
        
        if len(checkpoints) > 1:
            checkpoints.sort(key=os.path.getmtime)
            for old_file in checkpoints[:-1]:
                os.remove(old_file)
                print(f"Removed old checkpoint: {os.path.basename(old_file)}")


def setup_logging(experiment_tag, resume=False):
    """Setup logging to file and console"""
    log_file = f"checkpoints/{experiment_tag}-log.txt"
    
    # Create or append to log file
    mode = 'a' if resume else 'w'
    
    with open(log_file, mode) as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if resume:
            f.write(f"\n\n{'='*50}\n")
            f.write(f"RESUMED TRAINING - {timestamp}\n")
            f.write(f"{'='*50}\n")
        else:
            f.write(f"NEAT Snake Training Log - {timestamp}\n")
            f.write(f"Experiment Tag: {experiment_tag}\n")
            f.write(f"{'='*50}\n")
    
    return log_file


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def eval_genome(genome, config, generation_seeds=None):
    if generation_seeds is None:
        generation_seeds = [42, 123]
    
    total_fitness = 0
    
    for seed in generation_seeds:
        game = SnakeGame(grid_size=16)
        game.seed(seed)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        state = game.reset()
        
        grid_size = state['grid_size']
        food_eaten = 0
        max_steps = grid_size * grid_size * 2
        steps_taken = 0
        distance_improvements = 0
        close_approaches = 0
        prev_distance = None
        close_distance_threshold = max(1, grid_size // 8)
        
        positions_since_food = set()
        loop_penalty = 0
        
        for step in range(max_steps):
            food_pos = state.get('food', None)
            head = state['snake'][0]
            
            if head in positions_since_food:
                loop_penalty += 5
            
            positions_since_food.add(head)
            
            if food_pos is not None:
                current_distance = manhattan_distance(head, food_pos)
                
                if current_distance <= close_distance_threshold:
                    close_approaches += 1
                
                if prev_distance is not None and current_distance < prev_distance:
                    distance_improvements += 1
                    
                prev_distance = current_distance
                
            features = NEATAgent.extract_features(state, grid_size=grid_size)
            action = np.argmax(net.activate(features))
            state, reward, done = game.step(action)
            steps_taken += 1
            
            if reward == 1:
                food_eaten += 1
                prev_distance = None
                positions_since_food.clear()
            
            if done:
                if state.get('done_type') == 'illegal_action':
                    return 0  # Immediate zero fitness, exit function
                break
        
        food_score = food_eaten * 1000
        approach_bonus = distance_improvements * 5
        max_proximity_bonus = max_steps // 10
        proximity_bonus = min(close_approaches * 3, max_proximity_bonus)
        max_survival_bonus = grid_size // 2
        survival_bonus = min(steps_taken * 0.05, max_survival_bonus)
        
        game_fitness = food_score + approach_bonus + proximity_bonus + survival_bonus - loop_penalty
        total_fitness += game_fitness
    
    return total_fitness / len(generation_seeds)


def generate_generation_seeds(generation, games_per_genome):
    """Generate consistent seeds for a generation"""
    base_seed = 1000 + generation * 100
    return [base_seed + i * 17 for i in range(games_per_genome)]


def eval_genome_wrapper(genome_config_tuple):
    """Wrapper function for multiprocessing - takes tuple and returns (genome_id, fitness)"""
    genome_id, genome, config, generation_seeds = genome_config_tuple
    fitness = eval_genome(genome, config, generation_seeds)
    return genome_id, fitness


def evaluate_genomes(genomes, config, num_processes=None, generation=0, games_per_genome=2):
    """Parallel genome evaluation with fixed seeds per generation"""
    if num_processes is None:
        num_processes = mp.cpu_count()
    
    # Generate consistent seeds for this generation
    generation_seeds = generate_generation_seeds(generation, games_per_genome)
    
    # Prepare data for multiprocessing
    genome_data = [(genome_id, genome, config, generation_seeds) for genome_id, genome in genomes]
    
    # Use multiprocessing to evaluate genomes in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(eval_genome_wrapper, genome_data)
    
    # Assign fitness values back to genomes
    for genome_id, fitness in results:
        for gid, genome in genomes:
            if gid == genome_id:
                genome.fitness = fitness
                break


def clean_old_checkpoints(experiment_tag):
    """Keep only the latest checkpoint - used for final cleanup if needed"""
    pattern = f"checkpoints/{experiment_tag}-checkpoint*"
    checkpoints = glob.glob(pattern)
    
    if len(checkpoints) > 1:
        checkpoints.sort(key=os.path.getmtime)
        for old_file in checkpoints[:-1]:
            os.remove(old_file)
            print(f"Removed old checkpoint: {os.path.basename(old_file)}")


class GenerationTracker:
    """Helper class to track generation numbers for consistent seeding"""
    def __init__(self):
        self.current_generation = 0
    
    def get_eval_function(self, num_processes, games_per_genome):
        def eval_func(genomes, config):
            evaluate_genomes(genomes, config, num_processes, self.current_generation, games_per_genome)
            self.current_generation += 1
        return eval_func


def run_neat(config_path, n=10, path="checkpoints/best_neat_genome.pkl", checkpoint_interval=2, experiment_tag="default", num_processes=None, games_per_genome=2):
    log_file = setup_logging(experiment_tag, resume=False)
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    population = neat.Population(config)
    
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(LoggingReporter(log_file))
    population.add_reporter(CleaningCheckpointer(checkpoint_interval, filename_prefix=f"checkpoints/{experiment_tag}-checkpoint", experiment_tag=experiment_tag))
    population.add_reporter(BestGenomeSaver(path, log_file))
    population.add_reporter(LatestGenomeSaver(experiment_tag, log_file))


    # Create generation tracker for consistent seeding
    gen_tracker = GenerationTracker()
    eval_function = gen_tracker.get_eval_function(num_processes, games_per_genome)
    
    with open(log_file, 'a') as f:
        f.write(f"Starting training with {num_processes or mp.cpu_count()} processes\n")
        f.write(f"Target generations: {n}, Checkpoint interval: {checkpoint_interval}\n")
        f.write(f"Games per genome: {games_per_genome}\n\n")
    
    winner = population.run(eval_function, n=n)
    
    with open(path, "wb") as f:
        pickle.dump(winner, f)
    
    final_message = f"Training complete. Final fitness: {winner.fitness}"
    print(final_message)
    
    with open(log_file, 'a') as f:
        f.write(f"\n{final_message}\n")
        f.write(f"Training ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return winner


def run_neat_from_checkpoint(n=10, path="checkpoints/best_neat_genome.pkl", checkpoint_interval=2, experiment_tag="default", num_processes=None, games_per_genome=2):
    log_file = setup_logging(experiment_tag, resume=True)
    
    # Fixed pattern to match the actual checkpoint naming convention
    pattern = f"checkpoints/{experiment_tag}-checkpoint*"
    checkpoints = glob.glob(pattern)
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found with pattern: {pattern}")
    
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    resume_message = f"Resuming from: {os.path.basename(latest_checkpoint)}"
    print(resume_message)
    
    with open(log_file, 'a') as f:
        f.write(f"{resume_message}\n")
        f.write(f"Target additional generations: {n}, Checkpoint interval: {checkpoint_interval}\n")
        f.write(f"Using {num_processes or mp.cpu_count()} processes\n")
        f.write(f"Games per genome: {games_per_genome}\n\n")
    
    population = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
    
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(LoggingReporter(log_file))
    population.add_reporter(CleaningCheckpointer(checkpoint_interval, filename_prefix=f"checkpoints/{experiment_tag}-checkpoint", experiment_tag=experiment_tag))
    population.add_reporter(BestGenomeSaver(path, log_file))
    population.add_reporter(LatestGenomeSaver(experiment_tag, log_file))

    
    # Extract generation number from checkpoint to continue seeding properly
    checkpoint_gen = population.generation
    gen_tracker = GenerationTracker()
    gen_tracker.current_generation = checkpoint_gen
    eval_function = gen_tracker.get_eval_function(num_processes, games_per_genome)
    
    winner = population.run(eval_function, n=n)
    
    with open(path, "wb") as f:
        pickle.dump(winner, f)
    
    final_message = f"Training complete. Final fitness: {winner.fitness}"
    print(final_message)
    
    with open(log_file, 'a') as f:
        f.write(f"\n{final_message}\n")
        f.write(f"Training ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return winner


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="neat_config.txt")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--checkpoint_interval", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--processes", type=int, default=None, help="Number of processes for parallel evaluation (default: all CPU cores)")
    parser.add_argument("--games", type=int, default=2, help="Number of games per genome per generation (default: 2)")
    args = parser.parse_args()
    
    if args.path is None:
        args.path = f"checkpoints/{args.tag}-best.pkl"
    
    print(f"Tag: {args.tag}, Generations: {args.n}, Interval: {args.checkpoint_interval}")
    print(f"Using {args.processes or mp.cpu_count()} processes for parallel evaluation")
    print(f"Games per genome: {args.games}")
    
    if args.resume:
        run_neat_from_checkpoint(args.n, args.path, args.checkpoint_interval, args.tag, args.processes, args.games)
    else:
        run_neat(args.config, args.n, args.path, args.checkpoint_interval, args.tag, args.processes, args.games)