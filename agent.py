import numpy as np
import random
from collections import namedtuple, deque

from brain import QNetwork  # Import the brain we just built

import torch
import torch.nn.functional as F
import torch.optim as optim

# Check if GPU is available (makes training 10x faster)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, config, seed=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed if seed is not None else self.config.project.seed)

        # Get active environment config
        active_env_name = self.config.active_env
        env_config = self.config.environments[active_env_name]

        # --- Agent Parameters from Config ---
        self.DQN_type = self.config.agent.DQN_type
        self.use_replay_buffer = self.config.agent.get('use_replay_buffer', True)
        self.use_target_network = self.config.agent.get('use_target_network', True)
        self.buffer_size = self.config.agent.buffer_size
        self.batch_size = self.config.agent.batch_size
        self.gamma = self.config.agent.gamma
        self.lr = self.config.agent.lr
        self.update_every = self.config.agent.update_every

        # Q-Network (The "Local" brain that learns constantly)
        self.qnetwork_local = QNetwork(state_size, action_size, env_config.network.hidden_size,
                                       seed if seed is not None else self.config.project.seed,
                                       is_dueling=config.agent.get('is_dueling', False)).to(device)
        
        # Q-Network (The "Target" brain that stays stable)
        self.qnetwork_target = QNetwork(state_size, action_size, env_config.network.hidden_size,
                                       seed if seed is not None else self.config.project.seed,
                                       is_dueling=config.agent.get('is_dueling', False)).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory (We will define this class later)
        _buffer_size = self.buffer_size if self.use_replay_buffer else 1
        _batch_size = self.batch_size if self.use_replay_buffer else 1
        self.memory = ReplayBuffer(action_size, _buffer_size, _batch_size, seed if seed is not None else self.config.project.seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, tau):
        """
        Save experience in replay memory, and use random sample from buffer to learn.
        Params
        ======
            state (array_like): current state
            action (int): action taken
            reward (float): reward received
            next_state (array_like): next state
            done (bool): whether the episode has ended
            tau (float): interpolation parameter for soft update
        """
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.batch_size:
                experiences = self.memory.sample()
                q_val = self.learn(experiences, self.gamma, tau)
                return q_val   
        return None # Return None if we didn't learn on this step
    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Turn off training mode (we are just playing, not learning right now)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        # Turn training mode back on
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) # Pick Best
        else:
            return random.choice(np.arange(self.action_size))  # Pick Random

    def learn(self, experiences, gamma, tau):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            tau (float): interpolation parameter for soft update
        """
        states, actions, rewards, next_states, dones = experiences

        # Determine the network to use for evaluating the next state's value
        # For true "no target network", we use the local network itself.
        eval_network = self.qnetwork_local if not self.use_target_network else self.qnetwork_target

        # ------------------- update local network ------------------- #
        
        # 1. Get Q values for next states from target model:
        match self.DQN_type:
            case "DQN":
                # standard DQN:
                Q_targets_next = eval_network(next_states).detach().max(1)[0].unsqueeze(1)
                # detach - to prevent backpropagation through the target network, since we only want to update the local network right now
                # max(1)[0] - to get the maximum Q value for each next state across all possible actions (the [0] is because max returns a tuple of (values, indices))
                # unsqueeze(1) - to add an extra dimension so that Q_targets_next has the same shape as rewards (which is [batch_size, 1]) for the next step of computing Q targets.
            case "DDQN":
                # Double DQN:
                next_action = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
                Q_targets_next = eval_network(next_states).detach().gather(1, next_action)

        # 2. Compute Q targets for current states 
        # Formula: reward + (gamma * next_value)
        # If 'done' is 1 (game over), there is no next value, so we multiply by (1 - done)
        # we use the target network to compute the next value to keep the learning stable
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # 3. Get expected Q values from local model
        # gather(1, actions) extracts the Q-value for the specific action we actually took
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # 4. Compute loss (MSE: Mean Squared Error)
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # 5. Minimize the loss (Backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.use_target_network:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

        return Q_targets.detach().mean().item()  # return Q for distribution research          

    def update_lr(self, lr):
        """Update learning rate for the optimizer"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object."""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)