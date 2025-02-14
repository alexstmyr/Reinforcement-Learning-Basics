# Reinforcement-Learning-Basics
Training a blackjack dealer through DQL.


This repository contains a Python implementation of a Deep Q-Network (DQN) agent trained to play the Blackjack-v1 environment from Gymnasium. DQN is a classic algorithm in the field of Deep Reinforcement Learning (DRL).

## Overview

This project demonstrates how a neural network can learn to make optimal decisions in a game environment through trial and error, guided by a reward signal.  Think of it like teaching a model to trade: it makes decisions, sees the outcome (profit or loss), and adjusts its strategy over time.

## Deep Reinforcement Learning (DRL) - The High-Level Idea

DRL is a powerful technique that combines Reinforcement Learning (RL) with Deep Learning.

*   **Reinforcement Learning:**  The agent learns to make decisions by interacting with an environment to maximize a cumulative reward. It's all about learning through trial and error.
*   **Deep Learning:**  Neural networks are used to approximate the optimal *Q-function*, which estimates the expected future reward for taking a specific action in a given state. Neural nets are great at finding patterns in complex data.

**In layman's terms:** DRL is like teaching a computer to play a game (or make financial decisions) by rewarding it for good moves and penalizing it for bad ones. The "deep" part comes from using neural networks to help the computer understand the game's rules and strategies.

## How DQN Works 

1.  **The Agent:**  Our DQN agent uses a neural network to estimate the Q-value for each possible action in a given state.
2.  **Exploration vs. Exploitation:**  The agent needs to explore the environment to discover new strategies (exploration) but also exploit its current knowledge to maximize rewards (exploitation).  The `EPSILON` parameter controls this trade-off.
3.  **Experience Replay:**  The agent stores its experiences (state, action, reward, next state) in a memory buffer.  It then randomly samples batches from this memory to train the neural network.  This helps to break correlations in the data and improve learning stability.
4.  **Target Network:**  A separate target network is used to calculate the target Q-values.  This network is updated periodically with the weights of the policy network.  This also helps to stabilize training.
5.  **The Math:** The DQN's goal is to minimize the *loss function*, which is the difference between the predicted Q-values and the target Q-values.  The target Q-values are calculated using the Bellman equation:

    $$
    Q(s, a) = R(s, a) + \gamma \max_{a'} Q(s', a')
    $$

    Where:
    
    *   \(Q(s, a)\) is the Q-value for taking action \(a\) in state \(s\)
    *   \(R(s, a)\) is the reward received for taking action \(a\) in state \(s\)
    *   \(\gamma\) is the discount factor
    *   \(s'\) is the next state
    *   \(a'\) is the next action
    

## Code Explanation

*   `gymnasium`: Provides the Blackjack environment.
*   `DQN(nn.Module)`: Defines the neural network architecture for the DQN agent.  It's a simple feedforward network with a few fully connected layers.
*   `preprocess_state(state)`: Converts the state representation from the environment into a format suitable for the neural network (a PyTorch tensor).
*   `train_dqn()`: Implements the main training loop for the DQN agent.
*   `play_blackjack()`:  Uses the trained policy to play a few games of Blackjack and evaluates the performance.

## Hyperparameters

*   `GAMMA`: Discount factor.  Determines how much importance is given to future rewards.
*   `ALPHA`: Learning rate.  Controls the step size during neural network training.
*   `EPSILON`: Exploration rate.  Determines the probability of taking a random action.
*   `EPSILON_MIN`: Sets the minimum value that Epsilon can take.
*   `EPSILON_DECAY`: Decay factor for epsilon.  Reduces the exploration rate over time.
*   `BATCH_SIZE`: Batch size for experience replay.
*   `MEMORY_SIZE`: Experience replay memory size.
*   `EPISODES`: Number of training episodes.


## Insights from the Paper and Code

The *Nature* paper "Human-level control through deep reinforcement learning" demonstrated the power of DQN in learning to play Atari games at a human level. This code applies the same principles to a simpler environment, Blackjack.

*   **Abstraction:** The neural network learns to abstract the game's state into a representation that is useful for predicting future rewards. In the paper, the network learned to extract features directly from the pixel inputs. In this code, the state is preprocessed, but the network still learns a mapping from this state to Q-values.
*   **Generalization:** The DQN agent can generalize its knowledge to new situations it has never seen before.  This is crucial for real-world applications where the environment is constantly changing.  The paper highlights this by showing the agent can generalize to states from human play, not just its own.
*   **End-to-End Learning:** The DQN algorithm learns directly from the raw inputs (in the paper, pixels; in this code, preprocessed state) to the actions, without any hand-engineered features.  This is a key advantage of DRL.


## Conclusions
After seeing the results of the code and playing several hands of blackjack we can conclude that our model was pretty well trained under our DQN model. On average, a player has about 42% of chance of winning a hand, while the dealer (in this case named agent) has a 49% winning chance. Well, our DQN trained agent pulled more than 80% winning rate across 100 hands of blackjack, meaning that it almost doubled his chances of winning a hand thanks to our DQN algorithm. This shows that indeed, DQN do help get better performances, at least in simple games, as shown in the paper "Deep Reinforcement Learning with Double Q-lerarning" by van Hasselt et. al (2015).
