import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Load dataset
def load_data(csv_file):
    try:
        data = pd.read_csv(csv_file)
        print("Dataset loaded successfully!")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Normalize column names (strip spaces, lowercase)
    data.columns = data.columns.str.strip().str.lower()
    print("Available Columns:", data.columns.tolist())

    required_columns = {'co2', 'co', 'hc', 'nox', 'pmx', 'fuel', 'electricity',
                        'noise', 'waiting', 'pos', 'speed', 'angle', 'x', 'y'}

    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return None  # Exit if critical columns are missing

    return data

# Preprocess dataset
def preprocess_data(data):
    features = ['co2', 'co', 'hc', 'nox', 'pmx', 'fuel', 'electricity', 'noise',
                'waiting', 'pos', 'speed', 'angle', 'x', 'y']
    X = data[features].values
    return X

# Deep Deterministic Policy Gradient (DDPG) components
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience replay buffer
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = deque(maxlen=max_size)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# MADDPG agent
class MADDPG:
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim, action_dim)
        self.target_actor = Actor(state_dim, action_dim)
        self.target_critic = Critic(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.gamma = gamma
        self.replay_buffer = ReplayBuffer()

    def update(self, batch_size=64):
        if len(self.replay_buffer.buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)

        # Critic update
        next_actions = self.target_actor(next_states).detach()
        target_q_values = self.target_critic(next_states, next_actions).detach()
        target_q_values = rewards + self.gamma * target_q_values

        critic_loss = nn.MSELoss()(self.critic(states, actions), target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

# Training function
def train_agent(agent, data, episodes=100):
    state = preprocess_data(data)
    for episode in range(episodes):
        total_reward = 0
        step_count = 0  # Track steps per episode

        for i in range(len(state) - 1):
            action = agent.actor(torch.tensor(state[i], dtype=torch.float32)).detach().numpy()
            next_state = state[i + 1]
            reward = -state[i][8]  # Using waiting time as negative reward
            agent.replay_buffer.add((state[i], action, reward, next_state))
            agent.update()
            total_reward += reward
            step_count += 1

        avg_reward = total_reward / step_count if step_count > 0 else 0
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}, Avg Reward = {avg_reward:.2f}, Steps = {step_count}")

# Run the training process
if __name__ == "__main__":
    data = load_data("emissions.csv")
    if data is not None:  # Ensure data is loaded properly
        state_dim = len(preprocess_data(data)[0])
        action_dim = 2  # Example: Traffic signal control
        maddpg_agent = MADDPG(state_dim, action_dim)
        train_agent(maddpg_agent, data)
    else:
        print("Error: Could not load data. Please check the dataset.")
