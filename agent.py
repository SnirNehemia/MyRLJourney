import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torchrl")

import numpy as np
import random
from collections import namedtuple, deque
import torch.distributions as distributions

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
        seed_val = seed if seed is not None else self.config.project.seed
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        self.seed = seed_val

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
                return np.argmax(action_values.cpu().detach().numpy())
            else:
                return random.choice(np.arange(self.action_size))
        elif strategy == 'boltzmann':
            # Boltzmann exploration
            temperature = max(exploration_param, 1e-8) # exploration_param is temperature, ensure not zero
            probs = F.softmax(action_values / temperature, dim=1).cpu().detach().numpy().squeeze()
            return np.random.choice(np.arange(self.action_size), p=probs)
        else:
            # Default to greedy for testing (exploration_param=0) or unknown strategies
            return np.argmax(action_values.cpu().detach().numpy())

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
        random.seed(seed)
        self.seed = seed

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


class A2CAgent:
    """Interacts with and learns from the environment using Advantage Actor-Critic."""

    def __init__(self, state_size, action_size, config, seed=None):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        seed_val = seed if seed is not None else self.config.project.seed
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        self.seed = seed_val

        active_env_name = self.config.active_env
        env_config = self.config.environments[active_env_name]
        self.is_continuous = env_config.get('is_continuous', False)

        class ActorCritic(torch.nn.Module):
            def __init__(self, state_size, action_size, hidden_size, is_continuous, share_network=False):
                super().__init__()
                self.is_continuous = is_continuous
                self.share_network = share_network
                
                if self.share_network:
                    # Shared Backbone
                    self.shared_fc1 = torch.nn.Linear(state_size, hidden_size[0])
                    self.shared_fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
                    
                    if self.is_continuous:
                        self.actor_mean = torch.nn.Linear(hidden_size[1], action_size)
                        self.actor_std = torch.nn.Parameter(torch.zeros(1, action_size))
                    else:
                        self.actor_out = torch.nn.Linear(hidden_size[1], action_size)
                    self.critic_out = torch.nn.Linear(hidden_size[1], 1)
                else:
                    # Actor Network (Policy)
                    self.actor_fc1 = torch.nn.Linear(state_size, hidden_size[0])
                    self.actor_fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
                    if self.is_continuous:
                        self.actor_mean = torch.nn.Linear(hidden_size[1], action_size)
                        self.actor_std = torch.nn.Parameter(torch.zeros(1, action_size))
                    else:
                        self.actor_out = torch.nn.Linear(hidden_size[1], action_size)

                    # Critic Network (Value)
                    self.critic_fc1 = torch.nn.Linear(state_size, hidden_size[0])
                    self.critic_fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
                    self.critic_out = torch.nn.Linear(hidden_size[1], 1)
                    
            def forward(self, state, return_activations=False):
                if self.share_network:
                    s1 = F.relu(self.shared_fc1(state))
                    s2 = F.relu(self.shared_fc2(s1))
                    
                    if self.is_continuous:
                        mean = torch.tanh(self.actor_mean(s2))
                        std = F.softplus(self.actor_std).expand_as(mean) + 1e-5
                        dist = distributions.Normal(mean, std)
                        actor_output_viz = mean
                    else:
                        logits = self.actor_out(s2)
                        dist = distributions.Categorical(logits=logits)
                        actor_output_viz = F.softmax(logits, dim=-1)
                    
                    state_value = self.critic_out(s2)
                    
                    if return_activations:
                        activations = {
                            'input': state.detach().cpu().numpy().squeeze(),
                            'h1': s1.detach().cpu().numpy().squeeze(),
                            'h2': s2.detach().cpu().numpy().squeeze(),
                            'actor': actor_output_viz.detach().cpu().numpy().squeeze(),
                            'critic': state_value.detach().cpu().numpy().squeeze()
                        }
                        return dist, state_value, activations
                else:
                    # Actor Pass
                    a1 = F.relu(self.actor_fc1(state))
                    a2 = F.relu(self.actor_fc2(a1))
                    
                    if self.is_continuous:
                        mean = torch.tanh(self.actor_mean(a2))
                        std = F.softplus(self.actor_std).expand_as(mean) + 1e-5
                        dist = distributions.Normal(mean, std)
                        actor_output_viz = mean # for visualization
                    else:
                        logits = self.actor_out(a2)
                        dist = distributions.Categorical(logits=logits)
                        actor_output_viz = F.softmax(logits, dim=-1)
                    
                    # Critic Pass
                    c1 = F.relu(self.critic_fc1(state))
                    c2 = F.relu(self.critic_fc2(c1))
                    state_value = self.critic_out(c2)

                    if return_activations:
                        # Detach all tensors for visualization to avoid holding onto graph
                        activations = {
                            'input': state.detach().cpu().numpy().squeeze(),
                            'h1': a1.detach().cpu().numpy().squeeze(), # visualizing actor path
                            'h2': a2.detach().cpu().numpy().squeeze(),
                            'actor': actor_output_viz.detach().cpu().numpy().squeeze(),
                            'critic': state_value.detach().cpu().numpy().squeeze()
                        }
                        return dist, state_value, activations
                    
                return dist, state_value
                
            def init_weights(self):
                """Apply orthogonal initialization to weights."""
                for m in self.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                        torch.nn.init.constant_(m.bias, 0.0)
                
                # Smaller gain for output layers to start with a near-uniform policy
                torch.nn.init.orthogonal_(self.critic_out.weight, gain=1.0)
                if self.is_continuous:
                    torch.nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
                else:
                    torch.nn.init.orthogonal_(self.actor_out.weight, gain=0.01)

        share_network = self.config.agent.get('share_network', False)
        self.network = ActorCritic(state_size, action_size, env_config.network.hidden_size, self.is_continuous, share_network).to(device)
        self.network.init_weights()
        # Separate optimizers: critic uses a lower LR for stable value estimation.
        actor_params = [p for n, p in self.network.named_parameters() if 'actor' in n]
        critic_params = [p for n, p in self.network.named_parameters() if 'critic' in n or 'shared' in n]
        self.actor_optimizer = optim.Adam(actor_params, lr=config.agent.get('lr', 0.001))
        self.critic_optimizer = optim.Adam(critic_params, lr=config.agent.get('critic_lr', 0.0003))
        self.gamma = config.agent.gamma
        self.entropy_weight = config.agent.get('entropy_weight_start', 0.01)

    def act(self, state, exploration_param=None):
        """Returns actions for given state as per current policy."""
        
        if state.ndim == 1:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            is_single = True
        else:
            state_tensor = torch.from_numpy(state).float().to(device)
            is_single = False
            
        self.network.eval()
        with torch.no_grad():
            dist, _ = self.network(state_tensor)
        self.network.train()
        
        if self.is_continuous:
            action = dist.mean if exploration_param == 0.0 else dist.sample()
            action = action.cpu().detach().numpy()
            return action.flatten() if is_single else action
        else:
            action = torch.argmax(dist.logits, dim=-1) if exploration_param == 0.0 else dist.sample()
            action = action.cpu().detach().numpy()
            return action.item() if is_single else action

    def update_lr(self, lr):
        """Update learning rate for the actor optimizer; critic LR stays fixed."""
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = lr

    def update_entropy_weight(self, entropy_weight):
        self.entropy_weight = entropy_weight

    def learn_from_batch(self, memory):
        """Update policy and value parameters using a batch of n-step experiences with GAE.

        Structured as three phases:
          1. Compute frozen TD-lambda return targets (no_grad).
          2. Train the critic K times against those fixed targets.
          3. Recompute advantages from the improved critic; update the actor once.

        Separating critic and actor updates means K>1 critic epochs are valid on-policy:
        the frozen targets and the single actor step preserve the on-policy constraint.
        """
        states, actions, rewards, next_states, dones = zip(*memory)

        # Convert lists to tensors (Shape: n_steps, num_envs, ...)
        states_tensor = torch.from_numpy(np.array(states)).float().to(device)
        if self.is_continuous:
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        else:
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.long).to(device)

        rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32).to(device)

        n_steps, num_envs = states_tensor.shape[0], states_tensor.shape[1]
        states_flat = states_tensor.view(n_steps * num_envs, -1)
        last_state_tensor = torch.from_numpy(next_states[-1]).float().to(device)

        gae_lambda = self.config.agent.get('gae_lambda', 0.95)
        critic_weight = self.config.agent.get('critic_loss_weight', 0.5)
        k_critic_epochs = self.config.agent.get('k_critic_epochs', 1)
        actor_params = [p for n, p in self.network.named_parameters() if 'actor' in n]
        critic_params = [p for n, p in self.network.named_parameters() if 'critic' in n or 'shared' in n]

        # --- Phase 1: Compute frozen GAE targets (returns) ---
        # All K critic epochs train against these fixed targets; they do not change.
        with torch.no_grad():
            _, values_init_flat = self.network(states_flat)
            values_init = values_init_flat.view(n_steps, num_envs)
            _, last_value_init = self.network(last_state_tensor)
            last_value_init = last_value_init.view(num_envs)

            all_values_init = torch.cat((values_init, last_value_init.unsqueeze(0)), dim=0)
            gae = 0
            adv_init = torch.zeros(n_steps, num_envs).to(device)
            for t in reversed(range(n_steps)):
                delta = rewards_tensor[t] + self.gamma * all_values_init[t+1] * (1 - dones_tensor[t]) - all_values_init[t]
                gae = delta + self.gamma * gae_lambda * gae * (1 - dones_tensor[t])
                adv_init[t] = gae
            returns = (adv_init + values_init).view(-1)  # frozen targets for all K epochs

        # --- Phase 2: K critic-only update epochs ---
        for _ in range(k_critic_epochs):
            _, values_flat = self.network(states_flat)
            critic_loss = F.smooth_l1_loss(values_flat.view(-1), returns)
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            (critic_weight * critic_loss).backward()
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, max_norm=1.0).item()
            self.critic_optimizer.step()

        # --- Phase 3: Single actor update using improved critic values ---
        # Fresh forward pass: dists carries gradient for log_prob; values are detached for GAE.
        dists, values_final_flat = self.network(states_flat)
        values_final = values_final_flat.detach().view(n_steps, num_envs)
        with torch.no_grad():
            _, last_value_final = self.network(last_state_tensor)
            last_value_final = last_value_final.view(num_envs)

        all_values_final = torch.cat((values_final, last_value_final.unsqueeze(0)), dim=0)
        gae = 0
        advantages = torch.zeros(n_steps, num_envs).to(device)
        for t in reversed(range(n_steps)):
            delta = rewards_tensor[t] + self.gamma * all_values_final[t+1] * (1 - dones_tensor[t]) - all_values_final[t]
            gae = delta + self.gamma * gae_lambda * gae * (1 - dones_tensor[t])
            advantages[t] = gae

        # Global normalization with a min-std guard.
        # Per-env normalization (dim=0) zeros out advantages when all envs are hovering
        # (all values nearly identical → tiny per-env std → amplified noise).
        # Skipping normalization when std is very small lets raw small-negative advantages
        # reach the actor, giving a weak but correct gradient signal.
        adv_std = advantages.std()
        if adv_std > 1e-3:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
        advantages = advantages.view(-1).clamp(-5.0, 5.0)

        # Explained variance after K critic epochs against the frozen returns.
        with torch.no_grad():
            ev = (1.0 - (returns - values_final_flat.detach().view(-1)).var() / (returns.var() + 1e-8)).item()

        if self.is_continuous:
            actions_flat = actions_tensor.view(n_steps * num_envs, -1)
        else:
            actions_flat = actions_tensor.view(-1)

        log_probs = dists.log_prob(actions_flat)
        if self.is_continuous:
            log_probs = log_probs.sum(dim=-1)
            entropy_loss = dists.entropy().sum(dim=-1).mean()
        else:
            entropy_loss = dists.entropy().mean()

        entropy_weight = self.entropy_weight
        actor_loss = -(log_probs * advantages.detach()).mean()
        # Critic already updated K times in Phase 2; no critic term here.
        loss = actor_loss - entropy_weight * entropy_loss

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0).item()
        self.actor_optimizer.step()

        return {
            'actor_loss':       actor_loss.item(),
            'critic_loss':      (critic_weight * critic_loss).item(),
            'entropy_loss':     (entropy_weight * entropy_loss).item(),
            'raw_entropy':      entropy_loss.item(),
            'entropy_weight':   entropy_weight,
            'ev':               ev,
            'actor_grad_norm':  actor_grad_norm,
            'critic_grad_norm': critic_grad_norm,
            'values_sample':    values_final_flat.detach().view(-1).cpu().numpy(),
            'returns_sample':   returns.detach().cpu().numpy(),
        }

class PPOAgent:
    """Proximal Policy Optimization with clipped surrogate objective.

    The core idea in one sentence: store log π_old at collection time, then run
    multiple gradient epochs guarded by clip(r, 1-ε, 1+ε) so the policy can't
    move too far in any single update batch.

    How this differs from A2CAgent:
    - act() also returns log_prob (and value) so the rollout loop can store them.
    - learn_from_batch receives (state, action, reward, next_state, done,
      old_log_prob, old_value) tuples — the extra two fields are what enables
      the ratio-based clipped objective.
    - Single optimizer for the combined actor+critic+entropy loss (no separate
      actor_optimizer / critic_optimizer like A2C).
    - Multiple PPO epochs with random minibatch shuffling per epoch.
    - Optional KL early-stop: if the policy drifts too far, abort remaining epochs.
    """

    def __init__(self, state_size, action_size, config, seed=None):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        seed_val = seed if seed is not None else self.config.project.seed
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        self.seed = seed_val

        active_env_name = self.config.active_env
        env_config = self.config.environments[active_env_name]
        self.is_continuous = env_config.get('is_continuous', False)

        # --- Action scaling ---
        # tanh maps raw network output → (-1, 1). Most envs have action bounds other than
        # exactly [-1, 1]. Pendulum-v1 is [-2, 2], for example. Without scaling, the agent
        # physically can't apply full torque, which starves the reward signal and kills learning.
        #
        # We query the actual Gymnasium env for its action space bounds and derive:
        #   action_scale = (high - low) / 2   → half-range of the action space
        #   action_bias  = (high + low) / 2   → center of the action space
        # Then: mean = tanh(raw) * scale + bias, which maps (-1,1) → (low, high) exactly.
        import gymnasium as _gym
        _tmp_env = _gym.make(active_env_name)
        if self.is_continuous:
            _high = _tmp_env.action_space.high.astype(np.float32)
            _low  = _tmp_env.action_space.low.astype(np.float32)
            # Guard against unbounded dims (inf); fall back to ±1 so tanh is safe.
            _high = np.where(np.isinf(_high),  1.0, _high)
            _low  = np.where(np.isinf(_low),  -1.0, _low)
            action_scale = ((_high - _low) / 2.0)   # Pendulum: [2.0]
            action_bias  = ((_high + _low) / 2.0)   # Pendulum: [0.0]
        else:
            action_scale = np.ones(action_size,  dtype=np.float32)
            action_bias  = np.zeros(action_size, dtype=np.float32)
        _tmp_env.close()

        # --- Network: shared backbone by default for PPO ---
        # A shared backbone means actor and critic see the same learned representation.
        # This is the standard PPO architecture. The two heads (actor_mean/actor_std
        # and critic_out) then specialize from that shared feature extractor.
        class ActorCritic(torch.nn.Module):
            def __init__(self, state_size, action_size, hidden_size, is_continuous,
                         share_network=True, action_scale=None, action_bias=None):
                super().__init__()
                self.is_continuous = is_continuous
                self.share_network = share_network

                # Store scale/bias as non-trainable buffers so they move to GPU with .to(device)
                # and are saved/loaded with state_dict correctly.
                if action_scale is not None:
                    self.register_buffer('action_scale',
                                         torch.FloatTensor(action_scale).unsqueeze(0))
                    self.register_buffer('action_bias',
                                         torch.FloatTensor(action_bias).unsqueeze(0))
                else:
                    self.register_buffer('action_scale', torch.ones(1, action_size))
                    self.register_buffer('action_bias',  torch.zeros(1, action_size))

                if self.share_network:
                    self.shared_fc1 = torch.nn.Linear(state_size, hidden_size[0])
                    self.shared_fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
                    if self.is_continuous:
                        self.actor_mean = torch.nn.Linear(hidden_size[1], action_size)
                        # Learned log-std (state-independent): one scalar per action dim.
                        # Initialized to 0 → std = softplus(0) ≈ 0.693 at start.
                        self.actor_log_std = torch.nn.Parameter(torch.zeros(1, action_size))
                    else:
                        self.actor_out = torch.nn.Linear(hidden_size[1], action_size)
                    self.critic_out = torch.nn.Linear(hidden_size[1], 1)
                else:
                    self.actor_fc1 = torch.nn.Linear(state_size, hidden_size[0])
                    self.actor_fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
                    if self.is_continuous:
                        self.actor_mean = torch.nn.Linear(hidden_size[1], action_size)
                        self.actor_log_std = torch.nn.Parameter(torch.zeros(1, action_size))
                    else:
                        self.actor_out = torch.nn.Linear(hidden_size[1], action_size)
                    self.critic_fc1 = torch.nn.Linear(state_size, hidden_size[0])
                    self.critic_fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
                    self.critic_out = torch.nn.Linear(hidden_size[1], 1)

            def forward(self, state):
                if self.share_network:
                    s1 = F.relu(self.shared_fc1(state))
                    s2 = F.relu(self.shared_fc2(s1))
                    if self.is_continuous:
                        # tanh squashes raw output to (-1, 1), then we shift/scale to (low, high).
                        # The distribution is parameterized in the *actual* action space,
                        # so log_probs and sampled actions are automatically consistent.
                        mean = torch.tanh(self.actor_mean(s2)) * self.action_scale + self.action_bias
                        # std is NOT scaled by action_scale — it stays in a normalized range.
                        # Why? If we scale std by 2 (Pendulum), the initial std ≈ 1.39.
                        # With mean up to ±2 and std=1.39, ~15% of samples land outside [-2,2]
                        # and get clipped by the env, but our log_prob uses the pre-clip value.
                        # That inconsistency biases the IS ratio. Unscaled std ≈ 0.69 keeps
                        # clipping under 0.5% and the log_probs consistent.
                        std = F.softplus(self.actor_log_std) + 1e-5
                        dist = distributions.Normal(mean, std)
                    else:
                        dist = distributions.Categorical(logits=self.actor_out(s2))
                    value = self.critic_out(s2)
                else:
                    a1 = F.relu(self.actor_fc1(state))
                    a2 = F.relu(self.actor_fc2(a1))
                    if self.is_continuous:
                        mean = torch.tanh(self.actor_mean(a2)) * self.action_scale + self.action_bias
                        std = F.softplus(self.actor_log_std) + 1e-5
                        dist = distributions.Normal(mean, std)
                    else:
                        dist = distributions.Categorical(logits=self.actor_out(a2))
                    c1 = F.relu(self.critic_fc1(state))
                    c2 = F.relu(self.critic_fc2(c1))
                    value = self.critic_out(c2)
                return dist, value

            def init_weights(self):
                """Orthogonal init: keeps activation variance stable through deep networks."""
                for m in self.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                        torch.nn.init.constant_(m.bias, 0.0)
                # Small gain for output layers: start with a near-uniform policy and a
                # near-zero value estimate so the agent explores before committing.
                torch.nn.init.orthogonal_(self.critic_out.weight, gain=1.0)
                if self.is_continuous:
                    torch.nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
                else:
                    torch.nn.init.orthogonal_(self.actor_out.weight, gain=0.01)

        share_network = self.config.agent.get('share_network', True)
        self.network = ActorCritic(
            state_size, action_size, env_config.network.hidden_size,
            self.is_continuous, share_network,
            action_scale=action_scale, action_bias=action_bias
        ).to(device)
        self.network.init_weights()

        # --- Optimizer setup ---
        # Two modes depending on share_network:
        #
        # share_network=True  → single optimizer for the combined actor+critic+entropy loss.
        #   Works when both heads share a backbone. Conceptually clean, but the large critic
        #   loss early in training dominates the shared layers and can corrupt actor updates.
        #
        # share_network=False → separate actor_optimizer and critic_optimizer, mirroring A2C.
        #   Lets us run K critic-only epochs (stabilizing V(s)) before the actor PPO phase.
        #   Higher critic LR accelerates value convergence; lower actor LR keeps policy stable.
        self.use_separate_optimizers = not share_network
        if self.use_separate_optimizers:
            actor_params  = [p for n, p in self.network.named_parameters() if 'actor' in n]
            critic_params = [p for n, p in self.network.named_parameters() if 'critic' in n]
            self.actor_optimizer  = optim.Adam(actor_params,  lr=config.agent.get('lr', 3e-4), eps=1e-5)
            self.critic_optimizer = optim.Adam(critic_params, lr=config.agent.get('critic_lr', 1e-3), eps=1e-5)
        else:
            self.optimizer = optim.Adam(self.network.parameters(),
                                        lr=config.agent.get('lr', 3e-4), eps=1e-5)

        self.gamma = config.agent.gamma
        self.entropy_weight = config.agent.get('entropy_weight_start', 0.0)

    def act(self, state, exploration_param=None, return_extra=False):
        """Sample an action from the current policy.

        Parameters
        ----------
        state : np.ndarray
            Shape (state_size,) for a single env, or (num_envs, state_size) for vectorized.
        exploration_param : float or None
            Pass 0.0 to use the deterministic policy mean (evaluation mode).
        return_extra : bool
            If True, also return (log_prob, value) as numpy scalars/arrays.
            Used during rollout collection — the caller stores these alongside
            (state, action, reward, next_state, done) for the PPO update.

        Returns
        -------
        action : np.ndarray or scalar
        (log_prob, value) : np.ndarray of shape (num_envs,) — only when return_extra=True
        """
        if state.ndim == 1:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            is_single = True
        else:
            state_tensor = torch.from_numpy(state).float().to(device)
            is_single = False

        self.network.eval()
        with torch.no_grad():
            dist, value = self.network(state_tensor)

            if self.is_continuous:
                # Use mean (greedy) for evaluation; sample from the distribution during training.
                action = dist.mean if exploration_param == 0.0 else dist.sample()
                # Sum log probs over action dimensions (independence assumption for factored Normal).
                # e.g., for a 3-dim action: log p(a) = log p(a1) + log p(a2) + log p(a3)
                log_prob = dist.log_prob(action).sum(dim=-1)
            else:
                action = torch.argmax(dist.logits, dim=-1) if exploration_param == 0.0 else dist.sample()
                log_prob = dist.log_prob(action)
        self.network.train()

        action_np   = action.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy()          # (num_envs,) or scalar
        value_np    = value.squeeze(-1).cpu().numpy() # (num_envs,) or scalar

        if is_single:
            action_np   = action_np.flatten() if self.is_continuous else int(action_np.item())
            log_prob_np = float(log_prob_np.item())
            value_np    = float(value_np.item())

        if return_extra:
            return action_np, log_prob_np, value_np
        return action_np

    def update_lr(self, lr):
        opt = self.actor_optimizer if self.use_separate_optimizers else self.optimizer
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    def update_entropy_weight(self, entropy_weight):
        self.entropy_weight = entropy_weight

    def learn_from_batch(self, memory):
        """PPO update with two-phase structure when share_network=False.

        Phase 1  (only when use_separate_optimizers=True):
          Run k_critic_epochs of pure critic updates against frozen GAE targets.
          Higher critic LR means V(s) converges quickly before the actor moves.
          After K epochs, recompute advantages using the improved critic — this is
          the same trick A2CAgent uses and it's critical for advantage quality.

        Phase 2 (always):
          Run ppo_epochs of actor updates with the PPO clipped surrogate objective.
          The clipped ratio prevents the policy from moving too far in any single epoch.

        When share_network=True (single optimizer):
          Only Phase 2 runs — combined actor+critic+entropy loss per minibatch.
          The actor gradient is small relative to the critic gradient early in training,
          so the first policy updates may be noisy. Avoid this mode for early-stage training.

        Parameters
        ----------
        memory : list of tuples
            Each tuple: (state, action, reward, next_state, done, old_log_prob, old_value)
        """
        states, actions, rewards, next_states, dones, old_log_probs, old_values = zip(*memory)

        states_tensor        = torch.from_numpy(np.array(states)).float().to(device)
        rewards_tensor       = torch.tensor(np.array(rewards),      dtype=torch.float32).to(device)
        dones_tensor         = torch.tensor(np.array(dones),        dtype=torch.float32).to(device)
        old_log_probs_tensor = torch.tensor(np.array(old_log_probs), dtype=torch.float32).to(device)
        old_values_tensor    = torch.tensor(np.array(old_values),   dtype=torch.float32).to(device)

        if self.is_continuous:
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        else:
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.long).to(device)

        n_steps, num_envs = states_tensor.shape[0], states_tensor.shape[1]
        batch_size = n_steps * num_envs

        states_flat        = states_tensor.view(batch_size, -1)
        old_log_probs_flat = old_log_probs_tensor.view(-1)
        old_values_flat    = old_values_tensor.view(-1)
        actions_flat = actions_tensor.view(batch_size, -1) if self.is_continuous \
                       else actions_tensor.view(-1)

        gae_lambda      = self.config.agent.get('gae_lambda', 0.95)
        clip_eps        = self.config.agent.get('ppo_clip_eps', 0.2)
        ppo_epochs      = self.config.agent.get('ppo_epochs', 10)
        k_critic_epochs = self.config.agent.get('k_critic_epochs', 0)  # critic-only warmup
        num_minibatches = self.config.agent.get('num_minibatches', 8)
        minibatch_size  = max(1, batch_size // num_minibatches)
        critic_weight   = self.config.agent.get('critic_loss_weight', 0.5)
        norm_adv        = self.config.agent.get('normalize_advantage', True)
        clip_value_loss = self.config.agent.get('clip_value_loss', False)
        target_kl       = self.config.agent.get('target_kl', 0.0)
        last_state_tensor = torch.from_numpy(next_states[-1]).float().to(device)

        # --- Helper: compute GAE advantages and TD-lambda returns -----------------
        def _compute_gae(values_init_2d, last_val_1d):
            all_v = torch.cat([values_init_2d, last_val_1d.unsqueeze(0)], dim=0)
            gae, adv = 0, torch.zeros(n_steps, num_envs, device=device)
            for t in reversed(range(n_steps)):
                delta = (rewards_tensor[t]
                         + self.gamma * all_v[t + 1] * (1 - dones_tensor[t])
                         - all_v[t])
                gae = delta + self.gamma * gae_lambda * gae * (1 - dones_tensor[t])
                adv[t] = gae
            rets = (adv + values_init_2d).view(-1)
            return rets, adv.view(-1)

        # --- Helper: one minibatch pass through actor head -----------------------
        def _actor_step(mb_s, mb_a, mb_old_lp, mb_adv):
            dists, _ = self.network(mb_s)
            if self.is_continuous:
                new_lp = dists.log_prob(mb_a).sum(dim=-1)
                entropy = dists.entropy().sum(dim=-1).mean()
            else:
                new_lp = dists.log_prob(mb_a)
                entropy = dists.entropy().mean()
            ratio = torch.exp(new_lp - mb_old_lp)
            with torch.no_grad():
                lr_ = new_lp - mb_old_lp
                kl = ((torch.exp(lr_) - 1) - lr_).mean().item()
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            a_loss = -torch.min(surr1, surr2).mean()
            return a_loss, entropy, kl

        # -----------------------------------------------------------------------
        # STEP 1 — Initial GAE computation from current (pre-update) critic.
        #
        # "Returns" are the TD-lambda targets for the critic.  They are frozen
        # at this point and never change across phases, which gives the critic
        # a stable regression target (moving targets cause divergence).
        #
        # GAE backward recursion:
        #   delta_t = r_t + γ·V(s_{t+1})·(1−done) − V(s_t)   ← 1-step TD error
        #   A_t     = delta_t + γλ·(1−done)·A_{t+1}           ← recursive telescoping
        #
        # λ=0: A_t = delta_t  (pure 1-step TD, low variance, high bias)
        # λ=1: A_t = G_t−V(s_t) (Monte Carlo, unbiased, high variance)
        # λ=0.95: ~17-step effective horizon, balanced bias/variance tradeoff.
        # -----------------------------------------------------------------------
        with torch.no_grad():
            _, v_init = self.network(states_flat)
            v_init = v_init.view(n_steps, num_envs)
            _, v_last = self.network(last_state_tensor)
            v_last = v_last.view(num_envs)
            returns, advantages_flat = _compute_gae(v_init, v_last)
        values_init_sample = v_init.detach()

        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0
        total_kl          = 0.0
        total_grad_norm   = 0.0
        n_actor_steps     = 0
        n_critic_steps    = 0
        epochs_done       = 0

        # -----------------------------------------------------------------------
        # STEP 2 — Critic-only warmup epochs (only when use_separate_optimizers).
        #
        # Why warmup BEFORE the actor moves?
        # Early in training, V(s) ≈ 0 everywhere. The first GAE computation gives
        # advantages ≈ r_t * 17 ≈ −150 for all states (undifferentiated).
        # Running K critic epochs first makes V(s) converge toward the true values
        # before the actor sees ANY gradient. The subsequent actor PPO phase then
        # uses MUCH better advantages (differentiating good vs bad states), so the
        # first policy gradient step is in the right direction.
        #
        # This mirrors A2C's Phase 1 / Phase 2 design.  The key difference from
        # A2C: after critic warmup we RECOMPUTE advantages from the improved critic,
        # then apply the PPO *clipped ratio* in the actor update — that's what makes
        # it PPO rather than just "A2C with more critic epochs".
        # -----------------------------------------------------------------------
        if self.use_separate_optimizers and k_critic_epochs > 0:
            critic_params = [p for n, p in self.network.named_parameters() if 'critic' in n]
            for _ in range(k_critic_epochs):
                perm = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, minibatch_size):
                    idx = perm[start: start + minibatch_size]
                    if len(idx) < 4:
                        continue
                    _, v_mb = self.network(states_flat[idx])
                    v_mb = v_mb.squeeze(-1)
                    if clip_value_loss:
                        v_cl = old_values_flat[idx] + torch.clamp(
                            v_mb - old_values_flat[idx], -clip_eps, clip_eps)
                        c_loss = 0.5 * torch.max(
                            (v_mb - returns[idx]).pow(2),
                            (v_cl  - returns[idx]).pow(2)).mean()
                    else:
                        c_loss = 0.5 * F.mse_loss(v_mb, returns[idx])
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    c_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_params, max_norm=0.5)
                    self.critic_optimizer.step()
                    total_critic_loss += c_loss.item()
                    n_critic_steps    += 1

            # Recompute GAE with the improved critic so the actor phase
            # sees higher-quality advantages.  (Returns stay frozen.)
            with torch.no_grad():
                _, v_improved = self.network(states_flat)
                v_improved = v_improved.view(n_steps, num_envs)
                _, v_last2  = self.network(last_state_tensor)
                v_last2     = v_last2.view(num_envs)
                _returns_unused, advantages_flat = _compute_gae(v_improved, v_last2)
                # (returns for critic regression stay fixed from STEP 1)

        # -----------------------------------------------------------------------
        # STEP 3 — PPO actor epochs (clipped surrogate objective).
        #
        # The probability ratio r(θ) = π_new(a|s)/π_old(a|s) is computed fresh each
        # epoch from the stored old_log_probs. After each gradient step θ changes,
        # so the ratio drifts away from 1. Clipping prevents it from going too far:
        #
        #   L_CLIP = E[min(r·A, clip(r, 1−ε, 1+ε)·A)]
        #
        # For positive A: caps the gain at (1+ε)·A — don't over-commit.
        # For negative A: floors the penalty at (1−ε)·A — don't over-avoid.
        #
        # Without this, running ppo_epochs >> 1 on the same data would collapse
        # the policy (A2C with K actor updates is unstable; PPO is not).
        # -----------------------------------------------------------------------
        actor_params  = [p for n, p in self.network.named_parameters() if 'actor' in n] \
                        if self.use_separate_optimizers else None
        critic_params2 = [p for n, p in self.network.named_parameters() if 'critic' in n] \
                         if self.use_separate_optimizers else None

        for epoch in range(ppo_epochs):
            perm = torch.randperm(batch_size, device=device)

            for start in range(0, batch_size, minibatch_size):
                idx  = perm[start: start + minibatch_size]
                if len(idx) < 4:
                    continue

                mb_adv = advantages_flat[idx]
                if norm_adv and mb_adv.std() > 1e-6:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                mb_adv = mb_adv.clamp(-5.0, 5.0)

                a_loss, entropy, kl = _actor_step(
                    states_flat[idx], actions_flat[idx],
                    old_log_probs_flat[idx], mb_adv)

                if self.use_separate_optimizers:
                    # Separate path: actor optimizer only — critic network untouched.
                    loss = a_loss - self.entropy_weight * entropy
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        actor_params, max_norm=0.5).item()
                    self.actor_optimizer.step()
                else:
                    # Shared path: combined loss updates both heads together.
                    _, v_mb = self.network(states_flat[idx])
                    v_mb = v_mb.squeeze(-1)
                    if clip_value_loss:
                        v_cl = old_values_flat[idx] + torch.clamp(
                            v_mb - old_values_flat[idx], -clip_eps, clip_eps)
                        c_loss = 0.5 * torch.max(
                            (v_mb - returns[idx]).pow(2),
                            (v_cl  - returns[idx]).pow(2)).mean()
                    else:
                        c_loss = 0.5 * F.mse_loss(v_mb, returns[idx])
                    loss = a_loss + critic_weight * c_loss - self.entropy_weight * entropy
                    self.optimizer.zero_grad()
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.network.parameters(), max_norm=0.5).item()
                    self.optimizer.step()
                    total_critic_loss += c_loss.item()
                    n_critic_steps    += 1

                total_actor_loss += a_loss.item()
                total_entropy    += entropy.item()
                total_kl         += kl
                total_grad_norm  += grad_norm
                n_actor_steps    += 1

            epochs_done += 1

            if target_kl and target_kl > 0.0:
                if total_kl / max(n_actor_steps, 1) > target_kl:
                    break

        with torch.no_grad():
            _, v_final = self.network(states_flat)
            ev = (1.0 - (returns - v_final.detach().squeeze(-1)).var()
                  / (returns.var() + 1e-8)).item()

        na = max(n_actor_steps, 1)
        nc = max(n_critic_steps, 1)
        return {
            'actor_loss':       total_actor_loss  / na,
            'critic_loss':      total_critic_loss / nc,
            'entropy_loss':     (self.entropy_weight * total_entropy) / na,
            'raw_entropy':      total_entropy / na,
            'entropy_weight':   self.entropy_weight,
            'ev':               ev,
            'approx_kl':        total_kl / na,
            'epochs_done':      epochs_done,
            'grad_norm':        total_grad_norm / na,
            'values_sample':    values_init_sample.view(-1).cpu().numpy(),
            'returns_sample':   returns.detach().cpu().numpy(),
        }


class ReinforceAgent:
    """Interacts with and learns from the environment using REINFORCE (Monte Carlo Policy Gradient)."""

    def __init__(self, state_size, action_size, config, seed=None):
        self.config = config
        self.state_size = state_size
        self.action_size = action_size
        seed_val = seed if seed is not None else self.config.project.seed
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        self.seed = seed_val

        active_env_name = self.config.active_env
        env_config = self.config.environments[active_env_name]
        self.is_continuous = env_config.get('is_continuous', False)

        class PolicyNetwork(torch.nn.Module):
            def __init__(self, state_size, action_size, hidden_size, is_continuous):
                super().__init__()
                self.is_continuous = is_continuous
                
                self.fc1 = torch.nn.Linear(state_size, hidden_size[0])
                self.fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
                
                if self.is_continuous:
                    self.actor_mean = torch.nn.Linear(hidden_size[1], action_size)
                    self.actor_std = torch.nn.Parameter(torch.zeros(1, action_size))
                else:
                    self.actor_out = torch.nn.Linear(hidden_size[1], action_size)
                    
            def forward(self, state, return_activations=False):
                a1 = F.relu(self.fc1(state))
                a2 = F.relu(self.fc2(a1))
                
                if self.is_continuous:
                    mean = torch.tanh(self.actor_mean(a2))
                    std = F.softplus(self.actor_std).expand_as(mean) + 1e-5
                    dist = distributions.Normal(mean, std)
                    actor_output_viz = mean
                else:
                    logits = self.actor_out(a2)
                    dist = distributions.Categorical(logits=logits)
                    actor_output_viz = F.softmax(logits, dim=-1)
                    
                if return_activations:
                    activations = {
                        'input': state.detach().cpu().numpy().squeeze(),
                        'h1': a1.detach().cpu().numpy().squeeze(),
                        'h2': a2.detach().cpu().numpy().squeeze(),
                        'actor': actor_output_viz.detach().cpu().numpy().squeeze()
                    }
                    return dist, activations
                return dist

            def init_weights(self):
                """Apply orthogonal initialization to weights."""
                for m in self.modules():
                    if isinstance(m, torch.nn.Linear):
                        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                        torch.nn.init.constant_(m.bias, 0.0)
                
                if self.is_continuous:
                    torch.nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
                else:
                    torch.nn.init.orthogonal_(self.actor_out.weight, gain=0.01)

        self.network = PolicyNetwork(state_size, action_size, env_config.network.hidden_size, self.is_continuous).to(device)
        self.network.init_weights()
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.agent.get('lr', 0.005))
        self.gamma = config.agent.gamma

    def act(self, state, exploration_param=None):
        """Returns actions for given state as per current policy."""
        if state.ndim == 1:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            is_single = True
        else:
            state_tensor = torch.from_numpy(state).float().to(device)
            is_single = False
            
        self.network.eval()
        with torch.no_grad():
            dist = self.network(state_tensor)
        self.network.train()
        
        if self.is_continuous:
            action = dist.mean if exploration_param == 0.0 else dist.sample()
            action = action.cpu().detach().numpy()
            return action.flatten() if is_single else action
        else:
            action = torch.argmax(dist.logits, dim=-1) if exploration_param == 0.0 else dist.sample()
            action = action.cpu().detach().numpy()
            return action.item() if is_single else action

    def update_lr(self, lr):
        """Update learning rate for the optimizer"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def learn_from_episode(self, memory):
        """Update policy parameters using a full episode of experiences."""
        states, actions, rewards = zip(*memory)

        states_tensor = torch.from_numpy(np.array(states)).float().to(device)
        if self.is_continuous:
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        else:
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.long).to(device)
            
        # Calculate discounted returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # Normalize returns
        if len(returns_tensor) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

        # Forward pass
        dists = self.network(states_tensor)
        
        # Calculate Actor Loss
        if self.is_continuous:
            log_probs = dists.log_prob(actions_tensor).sum(dim=-1)
        else:
            log_probs = dists.log_prob(actions_tensor)
            
        loss = -(log_probs * returns_tensor).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()