import os
import math
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buffer)


class DQNNet(nn.Module):
    def __init__(self, in_channels, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 96, 96)
            n_flatten = self.features(dummy).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        if x.max() > 1.0:
            x = x / 255.0
        return self.fc(self.features(x))


def select_action(net, state, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    else:
        state_v = torch.FloatTensor(state).to(device).unsqueeze(0)
        with torch.no_grad():
            q_vals = net(state_v)
        return int(q_vals.argmax().item())


def compute_td_loss(policy_net, target_net, batch, device, gamma):
    states, actions, rewards, next_states, dones = batch
    states_v = torch.FloatTensor(states).to(device)
    next_states_v = torch.FloatTensor(next_states).to(device)
    actions_v = torch.LongTensor(actions).to(device)
    rewards_v = torch.FloatTensor(rewards).to(device)
    dones_v = torch.ByteTensor(dones).to(device)

    q_values = policy_net(states_v)
    state_action_values = q_values.gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_q_values = policy_net(next_states_v)
        next_actions = next_q_values.max(1)[1]
        next_q_target_values = target_net(next_states_v)
        next_state_values = next_q_target_values.gather(
            1, next_actions.unsqueeze(-1)
        ).squeeze(-1)
        target_values = rewards_v + gamma * next_state_values * (1 - dones_v.float())

    return nn.MSELoss()(state_action_values, target_values)


def train_double_dqn(
    num_episodes=500,
    max_steps=1000,
    batch_size=32,
    gamma=0.99,
    lr=1e-4,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=100000,
    target_update_freq=1000,
    buffer_size=100000,
    save_path="../models/double_dqn_carracing/model.zip",
):
    env = gym.make("CarRacing-v3", continuous=False)
    n_actions = env.action_space.n
    in_channels = env.observation_space.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQNNet(in_channels, n_actions).to(device)
    target_net = DQNNet(in_channels, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_size)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rewards = []
    steps_done = 0

    for _ in tqdm(range(num_episodes), desc="Episodes"):
        obs, _ = env.reset()
        state = np.transpose(obs, (2, 0, 1))
        ep_reward = 0

        for _ in range(max_steps):
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(
                -steps_done / epsilon_decay
            )
            action = select_action(policy_net, state, epsilon, n_actions, device)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = np.transpose(next_obs, (2, 0, 1))

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            steps_done += 1

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_td_loss(policy_net, target_net, batch, device, gamma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        rewards.append(ep_reward)

    torch.save(policy_net.state_dict(), save_path)
    np.save("rewards.npy", np.array(rewards))
    return rewards


def evaluate_double_dqn(
    model_path="../models/double_dqn_carracing/model.zip",
    episodes=5,
    max_steps=1000,
):
    env = gym.make("CarRacing-v3", continuous=False)
    n_actions = env.action_space.n
    in_channels = env.observation_space.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQNNet(in_channels, n_actions).to(device)
    policy_net.load_state_dict(torch.load(model_path, map_location=device))
    policy_net.eval()

    total = []
    for _ in range(episodes):
        obs, _ = env.reset()
        state = np.transpose(obs, (2, 0, 1))
        ep_reward = 0
        for _ in range(max_steps):
            with torch.no_grad():
                qv = policy_net(torch.FloatTensor(state).to(device).unsqueeze(0))
            action = int(qv.argmax().item())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = np.transpose(obs, (2, 0, 1))
            ep_reward += reward
            if done:
                break
        total.append(ep_reward)

    return float(np.mean(total))
