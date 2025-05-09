from tqdm import tqdm

import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from dm_control import suite
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env
from gymnasium.utils.save_video import save_video
from gymnasium.envs.registration import register
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dmc import make_dmc_env

import numpy as np
import random
from collections import deque
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, x):
        return self.max_action * self.net(x)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    def forward(self, x, u):
        return self.net(torch.cat([x, u], dim=1))

class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer()
        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        with torch.no_grad():
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pth"))

    def load(self, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, "actor.pth"), map_location=device))
        self.critic.load_state_dict(torch.load(os.path.join(directory, "critic.pth"), map_location=device))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        print("Model loaded successfully.")


# 訓練主程式
def train_ddpg(env_name="humanoid-walk", episodes=5000):
    env = make_dmc_env(env_name, seed=42, flatten=True, use_pixels=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action)
    checkpoint_dir = f"Q3/checkpoints/ddpg_{env_name}"
    if os.path.exists(os.path.join(checkpoint_dir, "actor.pth")):
        agent.load(checkpoint_dir)

    noise_std = 0.2
    noise_decay = 0.9999
    rewards = []

    for ep in tqdm(range(episodes)):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1000):  # longer horizon
            action = agent.select_action(state)
            action = (action + np.random.normal(0, noise_std, size=action_dim)).clip(-max_action, max_action)
            next_state, reward, done, _, _ = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            agent.train()
            state = next_state
            ep_reward += reward
            if done:
                break

        # noise_std *= noise_decay

        if (ep+1) % 50 == 0:
            agent.save(checkpoint_dir)
        rewards.append(ep_reward)
        tqdm.write(f"Episode {ep}, Reward: {ep_reward:.2f}, Avg for Last 100 Episodes: {np.mean(rewards[-100:]):.2f}, Noise Std: {noise_std}")

    env.close()

if __name__ == "__main__":
    train_ddpg()
