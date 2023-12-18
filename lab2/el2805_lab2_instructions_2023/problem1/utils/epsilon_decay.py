import numpy as np
def decay(Z, k, epsilon_max, epsilon_min, mode='linear_decay'):
    if mode == 'linear_decay':
        temp = epsilon_max - (epsilon_max - epsilon_min) * k / (Z - 1)
        return np.maximum(epsilon_min, temp)

    elif mode == 'exponential_decay':
        base = epsilon_min/epsilon_max
        power = k/(Z-1)
        temp = epsilon_max * base ** power
        return np.maximum(epsilon_min, temp)
    else:
        raise ValueError('Wrong mode, not accepted')
        # print('wrong mode')