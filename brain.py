import torch
import torch.nn as nn
import torch.nn.functional as F
    
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, seed=0, hidden_size=64):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    