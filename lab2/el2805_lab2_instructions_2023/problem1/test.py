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
# # def running_average(x, N):
# #     ''' Function used to compute the running average
# #         of the last N elements of a vector x
# #     '''
# #     if len(x) >= N:
# #         y = np.copy(x)
# #         y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
# #     else:
# #         y = np.zeros_like(x)
# #     return y
# #
# # # x = [1,2,3,4,5]
# # # N = 2
# # # print(running_average(x, N))
# #
# # # print(states)
# #
# # # size = 30000
# # buffer = ReplayBuffer(30000)
# # buffer.fill(5000)
# # # print(buffer, 'buffer')
# #
# #
env = gym.make('LunarLander-v2')
# state, _ = env.reset()
dim_state = len(env.observation_space.high)  # State dimensionality
# # # print(dim_state,'dimension of state')
n_actions = env.action_space.n               # Number of available actions
# # print(n_actions,'n_actions')
# # # print(env.action_space) # action: 0, 1, 2, 3
# # action = env.action_space.sample()
# # # print(type(state), state, '\n') # state: 8-dimensional variable, tuple
# # # env.render()
# #
# # Q_network = Q_Network(input_size=dim_state, hidden_size=64, output_size=n_actions)
# # target_network = copy.deepcopy(Q_network)
# # # torch.tensor([state], requires_grad=False, dtype=torch.float32)
# # state_tensor = torch.tensor(state, dtype=torch.float32, requires_grad=False)
# # # feed the state tensor to a neural network
# # q_value = Q_network(state_tensor)
# # # print(q_value, 'output q value')
# # # print(q_value.max(1)[0], '!!!!!')
# # value, index = torch.max(q_value, 0)
# # # print(index.item())
# # # print(index.item(), value)
# # # action = 2
# # next_state, reward, done, _, _ = env.step(action)
# # # print(next_state, 'next_state')
# # # buffer.append((state, action, reward, next_state, done))
# # #
# # # Sample a random batch of size N from B
# # # print(buffer.sample(10), 'buffer')
# # states, actions, rewards, next_states, dones = zip(*buffer.sample(3))
# #
# # # print(type(states)) # tuple
# # print(states, 'states')
# # # print('\n',rewards, 'rewards')
# # # print('\n',dones, 'dones')
# # rewards = torch.tensor(np.array(rewards))
# # dones = torch.tensor(np.array(dones))
# #
# # next_states_tensor = torch.tensor(np.vstack(next_states), dtype=torch.float32, requires_grad=False)
# # q_values = target_network(next_states_tensor)
# # target_values = rewards + 0.99 * torch.max(q_values, dim=1).values * (~dones)
# # target_values = target_values.view(-1, 1)
# # # print(target_values)
# # # q_values = Q_network(states_tensor)
# # # print(q_values,'q_values')
# # # values, indices = torch.max(q_values, 0)
# # # print(values)
# #
# # # print(torch.max(q_values, dim=1).values, 'largest values')
# # # Given values from the user's images
# #
# # # q_values = torch.tensor([[-0.0033, -0.0644,  0.0119,  0.1468],
# # #         [-0.0069, -0.0487,  0.0413,  0.1996],
# # #         [-0.0173, -0.0457,  0.0024,  0.1530]], grad_fn=<AddmmBackward0>)
# #
# # gamma = 0.99  # Assuming a discount factor gamma
# # states_tensor = torch.tensor(np.vstack(states), dtype=torch.float32, requires_grad=True)
# # print(np.vstack(states))
# # output_q_values = Q_network(states_tensor)
# # # print(output_q_values, 'output q values')
# # # print(actions, 'actions')
# # actions_tensor = torch.tensor(actions)
# # # actions_tensor.view(-1,1)
# # # print(actions_tensor.view(-1,1))
# # result = output_q_values.gather(1, actions_tensor.view(-1,1))
# # # print(result)
# #
# #
#
# env = gym.make('LunarLander-v2')
# state, _ = env.reset()
# print(env.reset())
#
# env = gym.make('LunarLander-v2')
# state, _ = env.reset()
# print(state.shape, 'state')
# action = env.action_space.sample()
#
# next_state, reward, terminated, truncated, _ = env.step(action)
# print(next_state.shape, 'next state')
# done = truncated or terminated
# experience = (state, action, reward, next_state, done)

net = Q_Network(input_size=dim_state, hidden_size=64, output_size=n_actions)
net.load_state_dict(torch.load('neural-network-1.pth'))
summary(net, input_size=(dim_state,), output_size=n_actions)
