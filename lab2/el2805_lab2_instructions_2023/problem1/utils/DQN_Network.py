import torch
import torch.nn as nn
import torch.optim as optim

'''
1. Initialization: setup of neural network layers and hyperparameters.
2. Forward Pass: Computation of Q-values for given state inputs
3. Training: update weights based on loss between predicted Q-values and target Q-values
4. Save models.
'''
class Q_Network(nn.Module):
    # nn.module should be tensor transferred from the states
    # input_size is the state space size, output_size is the action space size
    # Initialization
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        super(Q_Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Neural network requirement: not more than 2 hidden layers, 8< neurons per layer <128
        # fully connected layer1
        self.fc1 = nn.Linear(input_size, hidden_size)
        # activation function
        self.relu = nn.ReLU()

        # fully connected layer2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # activation function
        self.relu = nn.ReLU()

        # fully connected layer3, no activation in the output layer
        self.fc3 = nn.Linear(hidden_size, output_size)

        # # Optimizer -> Adam optimizer
        # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, input):
        """Forward pass to compute Q-values for the given state."""
        x = self.relu(self.fc1(input))
        x = self.relu(self.fc2(x))
        output = self.fc3(x)
        return output
