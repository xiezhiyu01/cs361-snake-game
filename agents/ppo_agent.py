from stable_baselines3 import PPO
from agents.base_agent import BaseAgent

class PPOAgent(BaseAgent):
    def __init__(self, path="checkpoints/ppo_snake", grid_size=16):
        super().__init__(grid_size)
        self.model = PPO.load(path)

    def select_action(self, obs, state=None):
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

    def seed(self, seed):
        pass