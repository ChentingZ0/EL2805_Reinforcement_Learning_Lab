# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#


'''
This is the main function of Problem1, the procedure follows the pesudo code in
the instructions
'''
# Load packages

import numpy as np
import gym
import torch
import copy
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import RandomAgent
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from utils.DQN_Network import Q_Network
# from utils import running_average
from utils.DQN_buffer import ReplayBuffer
from utils.epsilon_decay import decay


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 100                             # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
learning_rate = 5e-4
epsilon_max = 0.99
epsilon_min = 0.05
Z = 0.93*N_episodes
# N is of the order 4 − 128
training_batch_N = 128

# we suggest a buffer size of the order 5000 − 30000
buffer_size = 30000
buffer_ini = 15000
# the target network is updated every C steps. C ≈ L/N
C = int(buffer_size / training_batch_N)

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Initialize buffer B with maximum size L and fill it with random experiences
buffer = ReplayBuffer(buffer_size)
buffer.fill(buffer_ini)

# Initialze Q-network and target network
Q_network = Q_Network(input_size=dim_state, hidden_size=64, output_size=n_actions)
target_network = copy.deepcopy(Q_network)
target_network.eval()  # Set target network to evaluation mode (no gradient computation)


### Training process
# Initialize the optimizer and loss function
optimizer = optim.Adam(Q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
# losses = []
# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state, _ = env.reset()
    total_episode_reward = 0.
    t = 0
    step_count = 0  # counter
    epsilon = decay(Z=Z, k=i, epsilon_max=epsilon_max, epsilon_min=epsilon_min, mode='linear_decay')
    clipping_value = 1      # between 0.5-2
    while not done:
        # env.render()
        # convert the state to a pytorch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=False)
        # feed the state tensor to a neural network
        q_value = Q_network(state_tensor)
        max_q, max_action = torch.max(q_value, 0)
        max_q = max_q.item()
        max_action = max_action.item()
        # Take an action based on the epsilon-greedy policy
        if (random.random() < epsilon):
            # with probability ε, take a random action from action space
            action = env.action_space.sample()
        else:
            # with probability 1 - ε, select the action with the highest Q-value
            action = max_action

        # Observe next and append to buffer B
        next_state, reward, done, _, _ = env.step(action)
        # Update episode reward
        total_episode_reward += reward
        # Store experience in replay buffer
        buffer.append((state, action, reward, next_state, done))

        # Sample a random batch of size N from B
        batch = buffer.sample(training_batch_N)
        states, actions, rewards, next_states, dones = zip(*batch)  # reorganize

        states_tensor = torch.tensor(np.vstack(states), dtype=torch.float32, requires_grad=True)
        next_states_tensor = torch.tensor(np.vstack(next_states), dtype=torch.float32, requires_grad=False)
        q_values = target_network(next_states_tensor)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.bool)

        # target is a tensor of shape (batch_size, 1)
        target_values = rewards + discount_factor * torch.max(q_values, dim=1).values * (~dones)
        target_values = target_values.view(-1, 1)

        # Output q values
        q_values_temp = Q_network(states_tensor)
        actions_tensor = torch.tensor(actions, dtype=torch.int64)
        output_q_values = q_values_temp.gather(1, actions_tensor.view(-1, 1))

        # Update Q_network weights
        # Compute loss function
        loss = loss_fn(output_q_values, target_values)
        # losses.append(loss.item())
        # Initialize gradient to zero
        optimizer.zero_grad()
        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(Q_network.parameters(), clipping_value)

        # Perform backward pass (backpropagation)
        optimizer.step()

        state = next_state
        t += 1
        step_count += 1

        # Update the target network
        if step_count == C:
            target_network = copy.deepcopy(Q_network)
            step_count = 0      # empty the counter

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{} - loss: {:.2f}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1], loss.item()))

# Save the network
torch.save(Q_network, 'neural-network-1.pth')

# net = Q_Network(input_size=dim_state, hidden_size=64, output_size=n_actions)
# net.load_state_dict(torch.load('neural-network-1.pth'))
# summary(net, input_size=(dim_state,), output_size=n_actions)

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
# plt.show()

# plt.plot(losses)
# plt.title('Loss Function Over Time')
# plt.xlabel('Episode')
# plt.ylabel('Loss')
# plt.show()