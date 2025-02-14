import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

# Hyperparameters
GAMMA = 0.99  # Discount factor
ALPHA = 0.00025  # Learning rate
EPSILON = 1.0  # Exploration rate
EPSILON_MIN = 0.1  # Minimum epsilon value
EPSILON_DECAY = (EPSILON-EPSILON_MIN)/1000000  # Decay factor for epsilon
BATCH_SIZE = 32  # Batch size for experience replay
MEMORY_SIZE = 100  # Experience replay memory size
EPISODES = 10000  # Number of training episodes

# Create the Blackjack environment
env = gym.make('Blackjack-v1')

# Neural Network for Q-Learning
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Function to preprocess state (convert tuple to tensor)
def preprocess_state(state):
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0)

# Training loop
def train_dqn():
    global EPSILON
    
    input_dim = len(env.observation_space.spaces)  # (player sum, dealer card, usable ace)
    output_dim = env.action_space.n
    
    policy_net = DQN(input_dim, output_dim)
    target_net = DQN(input_dim, output_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=ALPHA)
    memory = deque(maxlen=MEMORY_SIZE)
    
    rewards_per_episode = []
    
    for episode in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Select action using epsilon-greedy
            if random.random() < EPSILON:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(policy_net(preprocess_state(state))).item()
            
            next_state, reward, done, _, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            # Training step
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                
                states, actions, rewards, next_states, dones = zip(*batch)
                states = torch.stack([preprocess_state(s).squeeze(0) for s in states])
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.stack([preprocess_state(s).squeeze(0) for s in next_states])
                dones = torch.tensor(dones, dtype=torch.bool)
                
                # Compute Q-values
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # Compute target Q-values
                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0]
                    target_q_values = rewards + (GAMMA * next_q_values * ~dones)
                
                # Compute loss and update weights
                loss = nn.MSELoss()(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        rewards_per_episode.append(total_reward)
        
        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
        
        # Update target network
        if episode % 500 == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Print progress
        if episode % 5000 == 0:
            print(f"Episode {episode}/{EPISODES}, Avg Reward: {np.mean(rewards_per_episode[-500:])}, Epsilon: {EPSILON:.4f}")
    
    # Smooth reward tracking for better visualization
    def moving_average(data, window_size=100):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(moving_average(rewards_per_episode, window_size=500), label="Smoothed Reward", color="blue")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward (Smoothed)")
    plt.title("DQL Agent Training Performance")
    plt.legend()
    plt.show()

    return policy_net

# Train the agent
trained_policy = train_dqn()

def play_blackjack(trained_policy, num_games=10):
    env = gym.make('Blackjack-v1', render_mode='human')
    score = []
    for game in range(num_games):
        state, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                action = torch.argmax(trained_policy(preprocess_state(state))).item()
            state, reward, done, _, _ = env.step(action)
        score += reward
        print(f"Game {game+1}: Reward {reward}")
        print(f'Score: {score}')
    env.close()

play_blackjack(trained_policy, num_games=10)
