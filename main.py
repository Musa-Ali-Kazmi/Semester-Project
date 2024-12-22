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
    def _init_(self, state_dim, action_dim, max_action):
        super(Actor, self)._init_()
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
    def _init_(self, state_dim, action_dim):
        super(Critic, self)._init_()
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
    def _init_(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def _len_(self):
        return len(self.buffer)

# Hyperparameters
ENV_NAME = "Walker2d-v5"
EPISODES = 10000
MAX_STEPS = 1000
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001
DISCOUNT_FACTOR = 0.99
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 1000000
TAU = 0.005  # For soft updates
EXPLORATION_NOISE = 0.05  # Stddev for action noise
POLICY_DELAY = 2  # Actor update delay
TARGET_NOISE = 0.2
NOISE_CLIP = 0.5

# Training Function
def train_td3():
    env = gym.make(ENV_NAME, forward_reward_weight=1.3, ctrl_cost_weight=0.004, healthy_reward=1.0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Create the models
    actor = Actor(state_dim, action_dim, max_action).to(device)
    target_actor = Actor(state_dim, action_dim, max_action).to(device)
    critic1 = Critic(state_dim, action_dim).to(device)
    target_critic1 = Critic(state_dim, action_dim).to(device)
    critic2 = Critic(state_dim, action_dim).to(device)
    target_critic2 = Critic(state_dim, action_dim).to(device)

    target_actor.load_state_dict(actor.state_dict())
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    last_episode = 0

    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE_ACTOR)
    critic1_optimizer = optim.Adam(critic1.parameters(), lr=LEARNING_RATE_CRITIC)
    critic2_optimizer = optim.Adam(critic2.parameters(), lr=LEARNING_RATE_CRITIC)
    replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    all_rewards = []
    all_actor_losses = []
    # all_critic1_losses = []
    total_steps = 0

    for episode in range(last_episode + 1, last_episode + EPISODES + 1, 1):
        state, _ = env.reset()
        total_reward = 0

        for _ in range(MAX_STEPS):
            total_steps += 1

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

                # Add noise to target actions
                noise = (torch.randn_like(actions) * TARGET_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP)
                next_actions = (target_actor(next_states) + noise).clamp(-max_action, max_action)

                # Compute target Q-values
                with torch.no_grad():
                    target_q1 = target_critic1(next_states, next_actions)
                    target_q2 = target_critic2(next_states, next_actions)
                    target_q = rewards + DISCOUNT_FACTOR * (1 - dones) * torch.min(target_q1, target_q2)

                # Update Critic 1
                current_q1 = critic1(states, actions)
                critic1_loss = nn.MSELoss()(current_q1, target_q)

                critic1_optimizer.zero_grad()
                critic1_loss.backward()
                critic1_optimizer.step()

                # Update Critic 2
                current_q2 = critic2(states, actions)
                critic2_loss = nn.MSELoss()(current_q2, target_q)

                critic2_optimizer.zero_grad()
                critic2_loss.backward()
                critic2_optimizer.step()

                # Delayed Actor Update
                if total_steps % POLICY_DELAY == 0:
                    actor_loss = -critic1(states, actor(states)).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    # Soft update target networks
                    for target_param, param in zip(target_actor.parameters(), actor.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                    for target_param, param in zip(target_critic1.parameters(), critic1.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

                    for target_param, param in zip(target_critic2.parameters(), critic2.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            if done:
                break

        all_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward}")
        # all_actor_losses.append(np.mean(actor_loss))

        # Save the models every 5000 episodes
        if (episode%5000 == 0):
            torch.save(actor.state_dict(), f"actor_episode_{episode}.pth")
            torch.save(critic1.state_dict(), f"critic1_episode_{episode}.pth")
            torch.save(critic2.state_dict(), f"critic2_episode_{episode}.pth")
            print(f"Saved models at episode {episode}")

            averaged_rewards = [sum(all_rewards[i:i+10]) / 10 for i in range(0, len(all_rewards), 10)]

            # Save the reward graph
            plt.figure()
            plt.plot(averaged_rewards)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Learning Curve")
            graph_filename = f"reward_graph_episode_{episode}.png"
            plt.savefig(graph_filename)
            plt.close()  # Close the plot to avoid memory issues
            print(f"Saved reward graph: {graph_filename}")

#             plt.figure()
#             plt.plot(all_actor_losses, marker='o', linestyle='-', color='g', label='Actor Loss')
#             # plt.plot(all_critic_losses, marker='o', linestyle='-', color='r', label='Critic Loss')
#             plt.xlabel("Episode")
#             plt.ylabel("Loss")
#             plt.title("Loss Curve")
#             plt.legend()
#             loss_graph_filename = f"loss_graph_episode_{episode}.png"
#             plt.savefig(loss_graph_filename)
#             plt.close()
#             print(f"Saved loss graph: {loss_graph_filename}")


    env.close()

    # Plot learning curve
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Learning Curve")
    plt.show()

if _name_ == "_main_":
    train_td3()



