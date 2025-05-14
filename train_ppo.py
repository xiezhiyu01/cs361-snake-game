import os
import numpy as np
import random
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym.wrappers import TimeLimit
from env.snake_gym_env import SnakeEnv

# Seeding
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# torch.manual_seed(SEED)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

# === IMPALA-style CNN ===
class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        in_channels = observation_space.shape[2]
        depths = [16, 32, 32]
        layers = []
        for depth in depths:
            layers.append(nn.Conv2d(in_channels, depth, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            in_channels = depth
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float().permute(0,3,1,2)
            n_flatten = self.cnn(sample).view(sample.shape[0], -1).shape[1]
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, obs):
        return self.linear(self.cnn(obs.permute(0, 3, 1, 2)))

# === Env factory ===
def make_env():
    def _init():
        env = SnakeEnv(grid_size=16, agent_name='ppo')
        env = TimeLimit(env, max_episode_steps=500)
        return env
    return _init

# === Vectorized Env ===
venv = DummyVecEnv([make_env() for _ in range(64)])
venv = VecMonitor(venv)

# === Policy kwargs ===
policy_kwargs = dict(
    features_extractor_class=ImpalaCNN,
    features_extractor_kwargs=dict(features_dim=256)
)

# === PPO Training ===
model = PPO(
    policy="CnnPolicy",
    env=venv,
    learning_rate=5e-4,
    n_steps=256,
    batch_size=512,               # 64 envs × 256 steps ÷ 8 minibatches
    n_epochs=3,
    gamma=0.999,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    seed=SEED,
    policy_kwargs=policy_kwargs
)

model.learn(total_timesteps=int(1e7))

# === Save Model ===
os.makedirs("checkpoints", exist_ok=True)
model.save("checkpoints/ppo_snake")
print("PPO model saved to checkpoints/ppo_snake.zip")
