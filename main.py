import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import os
import re

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))

# Define the Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
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
EPISODES = 2000
MAX_STEPS = 1000
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 1000000
TAU = 0.005  # For soft updates
EXPLORATION_NOISE = 0.1  # Stddev for action noise

# Training Function
def train_ddpg():
    env = gym.make(ENV_NAME, render_mode="human")  # Specify the render mode
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Create the models
    actor = Actor(state_dim, action_dim, max_action).to(device)
    target_actor = Actor(state_dim, action_dim, max_action).to(device)
    critic = Critic(state_dim, action_dim).to(device)
    target_critic = Critic(state_dim, action_dim).to(device)

    # Load the latest checkpoint if available
    checkpoint_pattern = re.compile(r"actor_episode_(\d+)\.pth")
    checkpoint_files = [f for f in os.listdir('.') if checkpoint_pattern.match(f)]

    last_episode = 0
    if checkpoint_files:
        # Get the checkpoint with the highest episode number
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(checkpoint_pattern.match(f).group(1)))
        last_episode = int(checkpoint_pattern.match(latest_checkpoint).group(1))

        actor_checkpoint = f"actor_episode_{last_episode}.pth"
        critic_checkpoint = f"critic_episode_{last_episode}.pth"

        if os.path.exists(actor_checkpoint) and os.path.exists(critic_checkpoint):
            actor.load_state_dict(torch.load(actor_checkpoint))
            target_actor.load_state_dict(actor.state_dict())
            critic.load_state_dict(torch.load(critic_checkpoint))
            target_critic.load_state_dict(critic.state_dict())
            print(f"Loaded saved models: {actor_checkpoint} and {critic_checkpoint}")
    else:
        target_actor.load_state_dict(actor.state_dict())
        target_critic.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE_ACTOR)
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE_CRITIC)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    all_rewards = []

    for episode in range(last_episode + 1, last_episode + EPISODES + 1, 1):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(MAX_STEPS):
            env.render()  # Render the environment

            # Select action with exploration noise 
            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
            action = actor(state_tensor).detach().cpu().numpy()[0]
            action += np.random.normal(0, EXPLORATION_NOISE, size=action_dim)
            action = np.clip(action, -max_action, max_action)

            # Step in the environment
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            # Training step
            if len(replay_buffer) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.FloatTensor(dones).to(device).unsqueeze(1)

                # Critic update
                with torch.no_grad():
                    next_actions = target_actor(next_states)
                    target_q = rewards + DISCOUNT_FACTOR * (1 - dones) * target_critic(next_states, next_actions)
                current_q = critic(states, actions)
                critic_loss = nn.MSELoss()(current_q, target_q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor update
                actor_loss = -critic(states, actor(states)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Target network updates
                for target_param, param in zip(target_critic.parameters(), critic.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                    target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            if done:
                break

        all_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")

        # Save the models every 500 episodes
        if (episode) % 500 == 0:
            torch.save(actor.state_dict(), f"actor_episode_{episode}.pth")
            torch.save(critic.state_dict(), f"critic_episode_{episode}.pth")
            print(f"Saved models at episode {episode}")

            # Save the reward graph
            plt.figure()
            plt.plot(all_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Learning Curve")
            graph_filename = f"reward_graph_episode_{episode}.png"
            plt.savefig(graph_filename)
            plt.close()  # Close the plot to avoid memory issues
            print(f"Saved reward graph: {graph_filename}")

    env.close()

    # Plot learning curve
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.show()

if __name__ == "__main__":
    train_ddpg()


























# # Define the Q-Network
# class QNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(QNetwork, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, action_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# # Replay Buffer
# class ReplayBuffer:
#     def __init__(self, capacity):
#         self.buffer = deque(maxlen=capacity)

#     def add(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         states, actions, rewards, next_states, dones = zip(*batch)
#         return (np.array(states), np.array(actions), np.array(rewards),
#                 np.array(next_states), np.array(dones))

#     def __len__(self):
#         return len(self.buffer)

# # Hyperparameters
# ENV_NAME = "Walker2d-v5"
# EPISODES = 500
# MAX_STEPS = 1000
# LEARNING_RATE = 0.001
# DISCOUNT_FACTOR = 0.99
# BATCH_SIZE = 64
# REPLAY_BUFFER_SIZE = 100000
# TARGET_UPDATE_FREQ = 1000
# EPSILON_START = 1.0
# EPSILON_END = 0.1
# EPSILON_DECAY = 5000

# # Training Function
# def train_ddqn():
#     env = gym.make(ENV_NAME)
#     state_dim = env.observation_space.shape[0]
#     action_dim = env.action_space.n

#     q_network = QNetwork(state_dim, action_dim)
#     target_network = QNetwork(state_dim, action_dim)
#     target_network.load_state_dict(q_network.state_dict())
#     target_network.eval()

#     optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
#     replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

#     epsilon = EPSILON_START
#     epsilon_decay = (EPSILON_START - EPSILON_END) / EPSILON_DECAY

#     all_rewards = []
#     steps = 0

#     for episode in range(EPISODES):
#         state, _ = env.reset()
#         total_reward = 0

#         for _ in range(MAX_STEPS):
#             env.render()  # Render the environment

#             steps += 1
#             if random.random() < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0)
#                 action = torch.argmax(q_network(state_tensor)).item()

#             next_state, reward, done, _, _ = env.step(action)
#             replay_buffer.add(state, action, reward, next_state, done)

#             state = next_state
#             total_reward += reward

#             if done:
#                 break

#             # Training step
#             if len(replay_buffer) >= BATCH_SIZE:
#                 states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)

#                 states = torch.FloatTensor(states)
#                 actions = torch.LongTensor(actions).unsqueeze(1)
#                 rewards = torch.FloatTensor(rewards).unsqueeze(1)
#                 next_states = torch.FloatTensor(next_states)
#                 dones = torch.FloatTensor(dones).unsqueeze(1)

#                 q_values = q_network(states).gather(1, actions)
#                 next_actions = torch.argmax(q_network(next_states), dim=1, keepdim=True)
#                 target_q_values = target_network(next_states).gather(1, next_actions)

#                 targets = rewards + (DISCOUNT_FACTOR * target_q_values * (1 - dones))
#                 loss = nn.MSELoss()(q_values, targets.detach())

#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#             # Update target network
#             if steps % TARGET_UPDATE_FREQ == 0:
#                 target_network.load_state_dict(q_network.state_dict())

#         # Decay epsilon
#         epsilon = max(EPSILON_END, epsilon - epsilon_decay)
#         all_rewards.append(total_reward)
#         print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

#     env.close()

#     # Plot learning curve
#     plt.plot(all_rewards)
#     plt.xlabel("Episode")
#     plt.ylabel("Total Reward")
#     plt.title("Learning Curve")
#     plt.show()

# if __name__ == "__main__":
#     train_ddqn()

