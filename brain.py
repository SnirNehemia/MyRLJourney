import torch
import torch.nn as nn
import torch.nn.functional as F
    
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, seed, is_dueling=False):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.is_dueling = is_dueling

        # Shared feature learning layers
        self.feature_layers = nn.ModuleList()
        self.feature_layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.feature_layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        
        # Output streams
        if self.is_dueling:
            # Value stream
            self.value_stream = nn.Linear(hidden_size[-1], 1)
            # Advantage stream
            self.advantage_stream = nn.Linear(hidden_size[-1], output_size)
        else:
            # Single output stream
            self.output_layer = nn.Linear(hidden_size[-1], output_size)

        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Pass through shared feature layers
        for layer in self.feature_layers:
            x = self.activation(layer(x))
        
        # Branch to appropriate streams
        if self.is_dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            # Combine value and advantage to get Q-values
            x = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            x = self.output_layer(x)
        
        return x
    