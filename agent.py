import numpy as np
import random
from collections import namedtuple, deque

from brain import QNetwork  # Import the brain we just built

import torch
import torch.nn.functional as F
import torch.optim as optim

from torchrl.data import LazyMemmapStorage, PrioritizedReplayBuffer, TensorDictPrioritizedReplayBuffer
from tensordict import TensorDict
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

        # PER parameters
        self.use_per = self.config.agent.get('use_per', False)
        if self.use_per:
            self.per_alpha = self.config.agent.get('per_alpha', 0.6)
            self.per_beta_start = self.config.agent.get('per_beta_start', 0.4)
            self.per_beta_end = self.config.agent.get('per_beta_end', 1.0)
            self.per_beta_frames = self.config.agent.get('per_beta_frames', 100000)
            self.per_beta = self.per_beta_start
            self.frame_count = 0
        # Q-Network (The "Local" brain that learns constantly)
        self.qnetwork_local = QNetwork(state_size, action_size, env_config.network.hidden_size,
                                       seed if seed is not None else self.config.project.seed,
                                       is_dueling=config.agent.get('is_dueling', False)).to(device)
        
        # Q-Network (The "Target" brain that stays stable)
        self.qnetwork_target = QNetwork(state_size, action_size, env_config.network.hidden_size,
                                       seed if seed is not None else self.config.project.seed,
                                       is_dueling=config.agent.get('is_dueling', False)).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)

        # Replay memory
        _buffer_size = self.buffer_size if self.use_replay_buffer else 1
        _batch_size = self.batch_size if self.use_replay_buffer else 1
        
        if self.use_per:
            self.memory = TensorDictPrioritizedReplayBuffer(
                alpha=self.per_alpha,
                beta=self.per_beta_start,
                priority_key="td_error",
                storage=LazyMemmapStorage(max_size=_buffer_size) # Define storage and max size
)
        else:
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
        if self.use_per:
            # torchrl's PrioritizedReplayBuffer expects TensorDict
            # It automatically sets initial priority to max_priority or 1.0
            exp = TensorDict({
                "state": torch.as_tensor(state, dtype=torch.float32),
                "action": torch.as_tensor(action, dtype=torch.long),
                "reward": torch.as_tensor(reward, dtype=torch.float32),
                "next_state": torch.as_tensor(next_state, dtype=torch.float32),
                "done": torch.as_tensor(done, dtype=torch.bool),
            }, batch_size=[]) 
            self.memory.add(exp)
            self.frame_count += 1
            # Anneal beta 
            self.per_beta = min(self.per_beta_end, self.per_beta_start + self.frame_count * (self.per_beta_end - self.per_beta_start) / self.per_beta_frames)
        else:
            self.memory.add(state, action, reward, next_state, done)


        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) >= self.batch_size:
                if self.use_per:
                    # Sample from PER buffer, get experiences, indices, and IS weights
                    sampled_data = self.memory.sample(batch_size=self.batch_size)
                    q_val, td_error = self.learn(sampled_data, self.gamma, tau, per_indices=sampled_data["index"], per_weights=sampled_data["priority_weight"])
                    # Update priorities in PER buffer
                    sampled_data["td_error"] = td_error.abs().squeeze().cpu().numpy()
                    self.memory.sampler._beta = self.per_beta # Update beta in the sampler
                    self.memory.update_priority(sampled_data["index"], sampled_data["td_error"] + 1e-6)
                    return q_val # TODO: shpuld I return the td-error?, td_error # Return both when PER is active
                else: # Not using PER
                    experiences = self.memory.sample()
                    q_val, _ = self.learn(experiences, self.gamma, tau) # _ is None here
                    return q_val # Only return q_val when PER is not active
        return None # Return None if we didn't learn on this step
    
    def act(self, state, exploration_param=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            exploration_param (float): exploration parameter (epsilon or temperature)
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        strategy = self.config.training.exploration.strategy

        if strategy == 'epsilon_greedy':
            # Epsilon-greedy action selection
            if random.random() > exploration_param: # exploration_param is epsilon
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))
        elif strategy == 'boltzmann':
            # Boltzmann exploration
            temperature = max(exploration_param, 1e-8) # exploration_param is temperature, ensure not zero
            probs = F.softmax(action_values / temperature, dim=1).cpu().data.numpy().squeeze()
            return np.random.choice(np.arange(self.action_size), p=probs)
        else:
            # Default to greedy for testing (exploration_param=0) or unknown strategies
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma, tau, per_indices=None, per_weights=None):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor] or TensorDict): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
            tau (float): interpolation parameter for soft update
            per_indices (torch.Tensor, optional): Indices of sampled experiences for PER.
            per_weights (torch.Tensor, optional): Importance Sampling weights for PER.
        """
        if self.use_per:
            states = experiences['state'].to(device)
            actions = experiences['action'].to(device).unsqueeze(-1) # actions need to be (batch_size, 1)
            rewards = experiences['reward'].to(device).unsqueeze(-1) # rewards need to be (batch_size, 1)
            next_states = experiences['next_state'].to(device)
            dones = experiences['done'].to(device).unsqueeze(-1) # dones need to be (batch_size, 1)
        else:
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
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones.float())) # .float() for boolean dones

        # 3. Get expected Q values from local model
        # gather(1, actions) extracts the Q-value for the specific action we actually took
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # 4. Compute loss (MSE: Mean Squared Error)
        td_error = Q_targets - Q_expected # Calculate TD error before loss for PER
        
        if self.use_per and per_weights is not None:
            # Apply Importance Sampling weights
            loss = (per_weights.to(device).unsqueeze(-1) * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean()
        else:
            loss = F.mse_loss(Q_expected, Q_targets)

        # 5. Minimize the loss (Backpropagation)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if self.use_target_network:
            self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

        if self.use_per:
            return Q_targets.detach().mean().item(), td_error.detach()
        else:
            return Q_targets.detach().mean().item(), None

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