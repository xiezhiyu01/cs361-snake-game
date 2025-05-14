import numpy as np
from tqdm import tqdm

def evaluate(agent, env, grid_size=16, episodes=10, seed_list=None, render=False, delay=0.1):
    scores = []
    steps = []

    for i in tqdm(range(episodes)):
        seed = seed_list[i] if seed_list else i
        env.seed(seed)
        agent.seed(seed)

        obs = env.reset()
        state = env.get_state()
        done = False
        score = 0

        while not done and state['steps'] < 1000:
            action = agent.select_action(obs, state)
            obs, _, done, info = env.step(action)
            state = env.get_state()

            score += info['reward']
            if render:
                env.render()
                import time
                time.sleep(delay)
            # print(f"Episode {i+1}/{episodes}, Action: {action}, Step: {state['steps']}, Score: {score}")

        scores.append(score)
        steps.append(state['steps'])

    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "all_scores": scores,
        "mean_steps": np.mean(steps),
        "std_steps": np.std(steps),
        "all_steps": steps
    }
