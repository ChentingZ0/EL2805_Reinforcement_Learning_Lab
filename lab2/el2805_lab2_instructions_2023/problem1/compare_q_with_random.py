import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from DQN_agent import RandomAgent
import pickle
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


def run_episode_randomly(env, episodes_num):
    agent = RandomAgent(n_actions=n_actions)
    episodes_reward_list_random = []
    for i in range(episodes_num):
        done = False
        total_reward = 0
        state, _ = env.reset()
        while not done:
            action = agent.forward(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = truncated or terminated
            total_reward += reward
            state = next_state
        episodes_reward_list_random.append(total_reward)
        env.close()
    return episodes_reward_list_random


with open('reward.pkl', 'rb') as file:
    episode_reward_list = pickle.load(file)

# Now my_list contains the restored list
# print(my_list)
env = gym.make('LunarLander-v2')
N_episodes = 500                             # Number of episodes
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions


rewards_trainedq_agent = running_average(episode_reward_list, n_ep_running_average)
rewards_random_agent = running_average(run_episode_randomly(env, episodes_num=N_episodes), n_ep_running_average)
fig = plt.figure()
plt.plot([i for i in range(1, N_episodes+1)], rewards_trainedq_agent, label='Avg. episode reward, Q-network agent')
plt.plot([i for i in range(1, N_episodes+1)], rewards_random_agent, label='Avg. episode reward, Random agent')
plt.xlabel('Episodes')
plt.ylabel('Total reward')
plt.title('Deep Q-Network vs Random Agent')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('./figures/Deep Q-Network vs Random Agent')
plt.show()
