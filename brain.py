import torch
import torch.nn as nn
import torch.nn.functional as F
    
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.fc.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.fc.append(nn.Linear(hidden_size[-1], output_size))
        self.activation = nn.ReLU()
    
    def forward(self, x):
        for layer in self.fc[:-1]:
            x = self.activation(layer(x))
        x = self.fc[-1](x)
        return x
    