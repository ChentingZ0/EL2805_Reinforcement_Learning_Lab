import numpy as np
import torch
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Load the trained Q-network model
model = torch.load('neural-network-1.pth')

# Define the range of y and ω values
y_values = np.linspace(0, 1.5, 100)
omega_values = np.linspace(-np.pi, np.pi, 100)

# Create a meshgrid for y and ω values
Y, Omega = np.meshgrid(y_values, omega_values)

# Initialize arrays to store the max Q-values and the corresponding actions
max_q_values = np.zeros_like(Y)
max_actions = np.zeros_like(Y)

# Evaluate the Q-network for each (y, ω) pair
for i in range(Y.shape[0]):
    for j in range(Y.shape[1]):
        # Construct the state s(y, ω) with the given restrictions
        state = np.array([0, Y[i, j], 0, 0, Omega[i, j], 0, 0, 0])

        # Convert state to a tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Get the Q-values from the model
        with torch.no_grad():
            q_values = model(state_tensor).numpy().flatten()

        # Store the max Q-value and the corresponding action
        max_q_values[i, j] = np.max(q_values)
        max_actions[i, j] = np.argmax(q_values)

# Plot the max Q-values
fig = plt.figure(figsize=(12, 6))

# First subplot for max Q-values
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(Y, Omega, max_q_values, cmap='viridis')
ax1.set_title('Max Q-values')
ax1.set_xlabel('y')
ax1.set_ylabel('ω')
ax1.set_zlabel('max Q')

# Second subplot for the actions
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(Y, Omega, max_actions, cmap='viridis')
ax2.set_title('Actions')
ax2.set_xlabel('y')
ax2.set_ylabel('ω')
ax2.set_zlabel('Action')

# Show plot
plt.show()

# Save the figures if needed
fig.savefig('./figures/q_values_and_actions.png')
