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