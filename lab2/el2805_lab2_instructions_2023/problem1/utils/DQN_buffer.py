import collections
import numpy as np
import gymnasium as gym

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = collections.deque(maxlen=size)
    
    def fill(self, length):
        env = gym.make('LunarLander-v2')
        state, _ = env.reset()
        for _ in range(length):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = truncated or terminated
            experience = (state, action, reward, next_state, done)
            self.buffer.append(experience)
            if done:
                state, _ = env.reset()
            else:
                state = next_state

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, length):
        index = np.random.choice(len(self.buffer), length, replace=False)
        result = []
        for i in index:
            result.append(self.buffer[i])
        return result
    
    def print(self):
        print(self.buffer)