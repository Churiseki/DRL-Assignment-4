import numpy as np
import random
from collections import deque
import os

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.max_action * torch.tanh(self.l3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)
    
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).float()
        self.actor_target = Actor(state_dim, action_dim, max_action).float()
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim).float()
        self.critic_target = Critic(state_dim, action_dim).float()
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = ReplayBuffer()
        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)

        # Critic loss
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (1 - done) * self.discount * target_Q
        current_Q = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q, target_Q.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pth"))

    def load(self, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, "actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, "critic.pth")))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        print("load successfully")

            
import gymnasium as gym
from tqdm import tqdm

env = gym.make("Pendulum-v1", render_mode="rgb_array")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPGAgent(state_dim, action_dim, max_action)
agent.load("Q1/checkpoints/ddpg_pendulum")
episodes = 8000
for ep in tqdm(range(episodes)):
    state, _ = env.reset()
    ep_reward = 0
    for t in range(200):
        action = agent.select_action(state)
        # exploration
        action = (action + np.random.normal(0, 0.1, size=action_dim)).clip(-max_action, max_action)
        next_state, reward, done, _, _ = env.step(action)
        agent.replay_buffer.push(state, action, reward, next_state, float(done))
        agent.train()

        state = next_state
        ep_reward += reward
    if ep % 50 == 0:
        agent.save("Q1/checkpoints/ddpg_pendulum")
    print(f"Episode {ep}, Reward: {ep_reward:.2f}")


import imageio

frames = []
state, _ = env.reset()
for _ in range(200):
    frame = env.render()
    frames.append(frame)
    action = agent.select_action(state)
    state, _, done, _, _ = env.step(action)

imageio.mimsave("pendulum.gif", frames, fps=30)
