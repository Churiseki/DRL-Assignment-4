import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from collections import deque
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
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


LOG_STD_MIN, LOG_STD_MAX = -20, 2


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, state):
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SACAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=2.5e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=2.5e-4)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=2.5e-4)
        self.target_entropy = -action_dim

        self.replay_buffer = ReplayBuffer()
        self.gamma = 0.99
        self.tau = 0.005
        self.max_action = max_action

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        if evaluate:
            mean, _ = self.actor(state)
            return (torch.tanh(mean) * self.max_action).cpu().data.numpy().flatten()
        else:
            action, _ = self.actor.sample(state)
            return action.cpu().data.numpy().flatten()

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - torch.exp(self.log_alpha) * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_new, log_prob = self.actor.sample(state)
        q1_new, q2_new = self.critic(state, action_new)
        actor_loss = (torch.exp(self.log_alpha) * log_prob - torch.min(q1_new, q2_new)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(directory, "actor.pth"))
        torch.save(self.critic.state_dict(), os.path.join(directory, "critic.pth"))

    def load(self, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, "actor.pth")))
        self.critic.load_state_dict(torch.load(os.path.join(directory, "critic.pth")))
        self.critic_target.load_state_dict(self.critic.state_dict())
        print("Loaded model successfully")


from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

env_name="humanoid-walk"
env = make_dmc_env(env_name, np.random.randint(0, 1000000), flatten=True, use_pixels=False)
# env.seed(0)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = SACAgent(state_dim, action_dim, max_action)
agent.load("checkpoints/sac_humanoid")
episodes = 10000
batch_size = 64
rewards = []
avg_rewards = []
for ep in tqdm(range(episodes)):
    state, _ = env.reset()
    ep_reward = 0
    for t in range(1000):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.replay_buffer.push(state.copy(), action, reward, next_state.copy(), float(done))
        if ep > 1000:
            agent.train(batch_size=batch_size)
        state = next_state
        ep_reward += reward
        if done:
            break

    if (ep+1) % 100 == 0:
        agent.save("checkpoints/sac_humanoid")

        plt.figure()
        plt.plot(avg_rewards, label="Reward")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title(f"Episode {ep+1} - Reward Trend")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Q3/fig/sac_humanoid_rewards_ep{ep+5001}.png")
        plt.close()
    rewards.append(ep_reward)
    avg_rewards.append(np.mean(rewards[-100:]))
    tqdm.write(f"Episode {ep}, Avg for Last 100 Ep: {np.mean(rewards[-100:]):.2f}, Reward: {ep_reward:.2f}")
