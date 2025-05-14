import argparse
from env.snake_gym_env import SnakeEnv
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.ppo_agent import PPOAgent
from runners.evaluate_agent import evaluate

def get_agent(grid_size, agent_name, model_path=None):
    if agent_name == "random":
        return RandomAgent(grid_size)
    elif agent_name == "greedy":
        return GreedyAgent(grid_size)
    elif agent_name == "ppo":
        return PPOAgent(model_path, grid_size)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="random", help="random | greedy | ppo")
    parser.add_argument("--model", type=str, default="checkpoints/ppo_snake", help="Path to the model")
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes (0 = skip)")
    parser.add_argument("--visualize", action="store_true", help="Visualize one episode")
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    agent = get_agent(args.grid, args.agent, args.model)
    env = SnakeEnv(grid_size=args.grid, agent_name=args.agent)

    print(f"Evaluating {args.agent} for {args.episodes} episodes...")

    result = evaluate(agent, 
                      env,
                      grid_size=args.grid, 
                      episodes=args.episodes,
                      render=args.visualize, 
                      delay=args.delay)

    print(f"Mean score: {result['mean_score']:.2f} ± {result['std_score']:.2f}")
    print(f"Mean steps: {result['mean_steps']:.2f} ± {result['std_steps']:.2f}")
    print(f"All scores: {result['all_scores']}")
    print(f"All steps: {result['all_steps']}")
    env.close()