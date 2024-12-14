import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# Hyperparameters
ENV_NAME = "Walker2d-v4"
EPISODES = 500
MAX_STEPS = 1000
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100000
TARGET_UPDATE_FREQ = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 5000

# Training Function
def train_ddqn():
    env = gym.make(ENV_NAME)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_network = QNetwork(state_dim, action_dim)
    target_network = QNetwork(state_dim, action_dim)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_START - EPSILON_END) / EPSILON_DECAY

    all_rewards = []
    steps = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(MAX_STEPS):
            env.render()  # Render the environment

            steps += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action = torch.argmax(q_network(state_tensor)).item()

            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

            # Training step
            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                states = torch.FloatTensor(states)
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                q_values = q_network(states).gather(1, actions)
                next_actions = torch.argmax(q_network(next_states), dim=1, keepdim=True)
                target_q_values = target_network(next_states).gather(1, next_actions)

                targets = rewards + (DISCOUNT_FACTOR * target_q_values * (1 - dones))
                loss = nn.MSELoss()(q_values, targets.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if steps % TARGET_UPDATE_FREQ == 0:
                target_network.load_state_dict(q_network.state_dict())

        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon - epsilon_decay)
        all_rewards.append(total_reward)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    env.close()

    # Plot learning curve
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.show()

if __name__ == "__main__":
    train_ddqn()
