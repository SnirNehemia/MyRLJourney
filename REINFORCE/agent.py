import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from REINFORCE.brain import Network  # Import the brain we just built

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config, seed=None):
        """Initialize a REINFORCE Agent object."""
        self.config = config
        self.state_size = state_size
        self.action_size = action_size

        # Get active environment config
        active_env_name = self.config.active_env
        env_config = self.config.environments[active_env_name]

        # --- Agent Parameters from Config ---
        self.gamma = self.config.agent.gamma
        self.lr = self.config.agent.lr

        # Use a proper seed
        _seed = seed if seed is not None else self.config.project.seed
        torch.manual_seed(_seed)

        # Policy Network
        self.policy_network = Network(state_size, action_size, env_config.network.hidden_size, _seed).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)

        # Episode-specific memory
        self.saved_log_probs = []
        self.rewards = []

    def act(self, state):
        """Returns an action for a given state as per current policy."""
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Get action probabilities from the policy network
        action_logits = self.policy_network(state)
        
        # Create a categorical distribution and sample an action
        m = Categorical(logits=action_logits)
        action = m.sample()
        
        # Store the log probability of the chosen action for the backward pass
        self.saved_log_probs.append(m.log_prob(action))
        
        return action.item()

    def finish_episode(self):
        """Called at the end of each episode to perform the learning step."""
        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted returns for each step, going backwards
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
            
        # Standardize returns for more stable training
        returns = torch.tensor(returns, device=device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        
        # Calculate the loss
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        # Update the policy network
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode memory
        del self.rewards[:]
        del self.saved_log_probs[:]