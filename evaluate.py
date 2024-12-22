import gymnasium as gym
import torch
import numpy as np
import re
import os
import matplotlib.pyplot as plt

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the Actor Network
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))

# Function to test the trained actor model
def test_ddpg(actor_checkpoint, env_name="Walker2d-v4", episodes=10, render=True):
    env = gym.make(env_name, render_mode="human" if render else None, terminate_when_unhealthy=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize the actor model
    actor = Actor(state_dim, action_dim, max_action).to(device)

    # Load the actor checkpoint
    if os.path.exists(actor_checkpoint):
        actor.load_state_dict(torch.load(actor_checkpoint, map_location=device))
        print(f"Loaded actor model from {actor_checkpoint}")
    else:
        raise FileNotFoundError(f"Checkpoint {actor_checkpoint} not found.")

    actor.eval()  # Set actor to evaluation mode

    rewards = []

    for episode in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0

        while True:
            if render:
                env.render()

            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
            with torch.no_grad():
                action = actor(state_tensor).cpu().numpy()[0]

            next_state, reward, done, _, _ = env.step(action)
            total_reward += reward

            state = next_state

            if done:
                break

        rewards.append(total_reward)
        print(f"Episode {episode}/{episodes}: Total Reward: {total_reward}")

    env.close()

    # Plot the rewards
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Evaluation Performance")
    plt.show()

if __name__ == "__main__":
    # Find the latest actor checkpoint
    checkpoint_pattern = re.compile(r"actor_episode_(\d+)\.pth")
    checkpoint_files = [f for f in os.listdir('.') if checkpoint_pattern.match(f)]

    if checkpoint_files:
        # Get the checkpoint with the highest episode number
        latest_checkpoint = max(checkpoint_files, key=lambda f: int(checkpoint_pattern.match(f).group(1)))
        print(f"Using checkpoint: {latest_checkpoint}")
        test_ddpg(latest_checkpoint, episodes=100, render=True)
    else:
        print("No actor checkpoints found. Please train the model first.")
