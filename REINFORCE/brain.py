import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, seed):
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Shared feature learning layers
        self.feature_layers = nn.ModuleList()
        self.feature_layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.feature_layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.output_layer = nn.Linear(hidden_size[-1], output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Pass through shared feature layers
        for layer in self.feature_layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)  
        return x
    