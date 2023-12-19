# Plot Rewards and steps

import numpy as np
import gym
import pickle
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
from utils.DQN_buffer import ReplayBuffer
from utils.epsilon_decay import decay
from utils.running_average import running_average

env = gym.make('LunarLander-v2')
env.reset()
# Parameters
N_episodes = 500                             # Number of episodes
discount_factor = 0.03                       # Value of the discount factor
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

import pickle

# Open the file containing the pickled list in binary read mode
with open('/reward.pkl', 'rb') as file:
    episode_reward_list = pickle.load(file)

# Now my_list contains the restored list
# print(my_list)


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
plt.savefig('./figures/training_3_gamma=0.03')
plt.show()


with open('reward.pkl', 'wb') as file:
    pickle.dump(episode_reward_list, file)
