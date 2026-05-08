import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time, importlib
from agent import Agent, A2CAgent, ReinforceAgent, device
from omegaconf import OmegaConf
import os, shutil

config = OmegaConf.load("config.yaml")

class DiscretizeBoxWrapper(gym.ActionWrapper):
    def __init__(self, env, bins):
        super().__init__(env)
        self.bins = bins
        self.action_dim = env.action_space.shape[0]
        self.action_space = gym.spaces.Discrete(bins ** self.action_dim)
        
    def action(self, action):
        continuous_action = np.zeros(self.action_dim, dtype=np.float32)
        curr_action = action
        for i in range(self.action_dim):
            bin_idx = curr_action % self.bins
            curr_action = curr_action // self.bins
            low = self.env.action_space.low[i]
            high = self.env.action_space.high[i]
            if np.isinf(low): low = -1.0
            if np.isinf(high): high = 1.0
            continuous_action[i] = low + (bin_idx / (self.bins - 1)) * (high - low)
        return continuous_action

def dqn(config, DQN_type=None, seed=None, record_name=None, n_episodes=None, run_type='train', study_name=None):
    """Deep Q-Learning training loop."""

    active_env_name = config.active_env
    env_config = config.environments[active_env_name]

    # Initialize Environment
    env_params = {}
    if 'lunar_params' in env_config:
        env_params = OmegaConf.to_container(env_config.lunar_params)
    env = gym.make(active_env_name, **env_params)
    
    if not env_config.get('is_continuous', False) and isinstance(env.action_space, gym.spaces.Box):
        bins = env_config.get('discrete_bins', 3)
        env = DiscretizeBoxWrapper(env, bins)

    # --- Fake Actions Logic ---
    if env_config.get('is_continuous', False):
        real_action_size = env.action_space.shape[0]
    else:
        real_action_size = env.action_space.n
        
    agent_action_size = real_action_size
    use_fake_actions = env_config.get('use_fake_actions', False)
    if use_fake_actions:
        num_fake = env_config.get('num_fake_actions', 0)
        agent_action_size += num_fake
        if num_fake > 0:
            print(f"INFO: Using {num_fake} fake actions. Agent action space: {agent_action_size}")

    agent_state_size = env.observation_space.shape[0]

    _version = config.project.version.replace(".", "-")
    _run_name = record_name if record_name is not None else config.save_parameters.run_name

    # Create a directory for the run
    if study_name is not None:
        output_dir = f"raw_results/{active_env_name}/{_version}/{run_type}/{study_name}/{_run_name}"
    else:
        output_dir = f"raw_results/{active_env_name}/{_version}/{run_type}/{_run_name}"
    os.makedirs(output_dir, exist_ok=True)
    # Save the exact config used for this run for reproducibility
    OmegaConf.save(config, os.path.join(output_dir, "run_config.yaml"))

    # Use config defaults unless overridden by experiment arguments
    # If an argument is passed, it overrides the config for this specific run.
    if DQN_type is not None:
        config.agent.DQN_type = DQN_type
    current_seed = seed if seed is not None else config.project.seed
    total_episodes = n_episodes if n_episodes is not None else config.training.n_episodes

    agent = Agent(state_size=agent_state_size, action_size=agent_action_size, config=config, seed=current_seed)
    
    scores = []
    scores_window = deque(maxlen=100)
    q_values_history = [] 
    avg_max_q_history = []

    # --- Exploration Parameters ---
    exploration_cfg = config.training.exploration
    strategy = exploration_cfg.strategy
    if strategy == 'epsilon_greedy':
        exploration_param = exploration_cfg.epsilon.start
        exploration_end = exploration_cfg.epsilon.end
        exploration_decay = exploration_cfg.epsilon.decay
    else: # boltzmann
        exploration_param = exploration_cfg.boltzmann.temperature_start
        exploration_end = exploration_cfg.boltzmann.temperature_end
        exploration_decay = exploration_cfg.boltzmann.temperature_decay

    lr = config.training.get('lr_start', config.agent.lr)
    tau = config.training.get('tau_start', config.agent.tau)
    agent.update_lr(lr)

    best_score = -float('inf')
    time_ref = time.time()

    for i_episode in range(1, total_episodes + 1):
        state, info = env.reset(seed=current_seed*total_episodes+i_episode)
        score = 0
        episode_q_vals = []
        episode_max_q_vals = []
        
        for t in range(config.training.max_t):
            # Get max predicted Q-value for the current state (for logging)
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            agent.qnetwork_local.eval()
            with torch.no_grad():
                action_values = agent.qnetwork_local(state_tensor)
            agent.qnetwork_local.train()
            episode_max_q_vals.append(torch.max(action_values).item())

            agent_action = agent.act(state, exploration_param)

            # Map agent action to real environment action
            env_action = agent_action
            if use_fake_actions and agent_action >= real_action_size:
                map_to_action = env_config.get('fake_action_maps_to', 0)
                env_action = map_to_action

            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            # --- Sparse Reward Logic for LunarLander ---
            is_lunar_lander_sparse = (active_env_name == "LunarLander-v3" and
                                      env_config.get('sparse_reward', False))
            if is_lunar_lander_sparse and not terminated:
                reward = 0.0
            
            # The agent.step call needs the original agent_action to learn correctly
            q_val = agent.step(state, agent_action, reward, next_state, done, tau)
            # # use this if i need to return td-error
            # if agent.use_per:
            #     q_val, _ = agent.step(state, agent_action, reward, next_state, done, tau)
            # else:
            #     q_val = agent.step(state, agent_action, reward, next_state, done, tau)
            if q_val is not None:
                episode_q_vals.append(q_val)
                
            state = next_state
            score += reward
            if done:
                break 
                
        scores_window.append(score)       
        scores.append(score)
        
        if len(episode_q_vals) > 0:
            q_values_history.append(np.mean(episode_q_vals))
        else:
            q_values_history.append(0 if len(q_values_history)==0 else q_values_history[-1])

        if len(episode_max_q_vals) > 0:
            avg_max_q_history.append(np.mean(episode_max_q_vals))
        else:
            avg_max_q_history.append(0 if len(avg_max_q_history)==0 else avg_max_q_history[-1])

        exploration_param = max(exploration_end, exploration_decay * exploration_param)
        lr = max(config.training.get('lr_end', 0.0), config.training.get('lr_decay', 1.0) * lr)
        tau = max(config.training.get('tau_end', config.agent.tau), config.training.get('tau_decay', 1.0) * tau)
        agent.update_lr(lr)

        current_avg = np.mean(scores_window)
        
        # Save best model based on moving average score
        if current_avg > best_score and i_episode > 100:
            best_score = current_avg
            torch.save(agent.qnetwork_local.state_dict(),
            f'{output_dir}/{_run_name}_local_best.pth')

        # Only print progress if we are running a standard training session (not an experiment)
        if n_episodes is None:
            print(f'\rEpisode {i_episode}\tAverage Score: {current_avg:.2f}', end="")
            if i_episode % 100 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {time.time() - time_ref:.0f} seconds = {(time.time() - time_ref)/60:.0f} minutes')
                
            if i_episode % 250 == 0:
                torch.save(agent.qnetwork_local.state_dict(),
                            f'{output_dir}/{_run_name}_{i_episode}-eps.pth')
                
            # Check Win Condition
            if current_avg >= env_config.win_condition:
                print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                # Save the trained neural network weights!
                torch.save(agent.qnetwork_local.state_dict(), f'{output_dir}/{_run_name}_best.pth')
                break
        else:
            if i_episode % 10 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {current_avg:.2f}\tTime: {time.time() - time_ref:.0f} seconds = {(time.time() - time_ref)/60:.0f} minutes', end="")
            if i_episode % 250 == 0:
                print(f'')

    # Pad results if training ended early to ensure consistent lengths for plotting
    if len(scores) < total_episodes:
        padding_needed = total_episodes - len(scores)
        if scores: # Check if scores is not empty
            scores.extend([scores[-1]] * padding_needed)
            q_values_history.extend([q_values_history[-1]] * padding_needed)
            avg_max_q_history.extend([avg_max_q_history[-1]] * padding_needed)
        else: # Handle case where no episodes were completed
            scores.extend([0] * padding_needed)
            q_values_history.extend([0] * padding_needed)
            avg_max_q_history.extend([0] * padding_needed)

    torch.save(agent.qnetwork_local.state_dict(), f'{output_dir}/{_run_name}_last.pth')    
    return scores, q_values_history, avg_max_q_history

def a2c(config, seed=None, record_name=None, n_episodes=None, run_type='train', study_name=None):
    """Advantage Actor-Critic (A2C) training loop."""

    active_env_name = config.active_env
    env_config = config.environments[active_env_name]

    num_envs = config.agent.get('num_envs', 8)

    def make_env():
        env_params = {}
        if 'lunar_params' in env_config:
            env_params = OmegaConf.to_container(env_config.lunar_params)
        e = gym.make(active_env_name, **env_params)
        if not env_config.get('is_continuous', False) and isinstance(e.action_space, gym.spaces.Box):
            bins = env_config.get('discrete_bins', 3)
            e = DiscretizeBoxWrapper(e, bins)
        return e

    # Initialize Vector Environment
    env = gym.vector.AsyncVectorEnv([make_env for _ in range(num_envs)])

    # --- Fake Actions Logic ---
    dummy_env = make_env()
    if env_config.get('is_continuous', False):
        real_action_size = dummy_env.action_space.shape[0]
    else:
        real_action_size = dummy_env.action_space.n
    dummy_env.close()

    agent_action_size = real_action_size
    use_fake_actions = env_config.get('use_fake_actions', False)
    if use_fake_actions:
        num_fake = env_config.get('num_fake_actions', 0)
        agent_action_size += num_fake

    agent_state_size = dummy_env.observation_space.shape[0]

    _version = config.project.version.replace(".", "-")
    _run_name = record_name if record_name is not None else config.save_parameters.run_name

    if study_name is not None:
        output_dir = f"raw_results/{active_env_name}/{_version}/{run_type}/{study_name}/{_run_name}"
    else:
        output_dir = f"raw_results/{active_env_name}/{_version}/{run_type}/{_run_name}"
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, "run_config.yaml"))

    current_seed = seed if seed is not None else config.project.seed
    total_episodes = n_episodes if n_episodes is not None else config.training.n_episodes

    agent = A2CAgent(state_size=agent_state_size, action_size=agent_action_size, config=config, seed=current_seed)
    
    lr = config.training.get('lr_start', config.agent.lr)
    agent.update_lr(lr)

    scores = []
    scores_window = deque(maxlen=100)
    avg_state_value_history = []

    best_score = -float('inf')
    time_ref = time.time()

    n_steps = config.agent.get('n_steps', 20)
    memory = []

    state, info = env.reset(seed=current_seed)
    episode_rewards = np.zeros(num_envs)
    
    completed_episodes = 0
    last_logged_episode = 0
    episodes_completed_this_batch = 0

    while completed_episodes < total_episodes:
        episode_state_vals = []
        
        # Collect n_steps of transitions across all parallel environments
        for t in range(n_steps):
            state_tensor = torch.from_numpy(state).float().to(device)
            agent.network.eval()
            with torch.no_grad():
                _, state_value = agent.network(state_tensor)
            agent.network.train()
            episode_state_vals.append(state_value.mean().item())

            agent_action = agent.act(state)

            env_action = agent_action.copy()
            if use_fake_actions and not env_config.get('is_continuous', False):
                map_to_action = env_config.get('fake_action_maps_to', 0)
                env_action[agent_action >= real_action_size] = map_to_action

            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated | truncated
                
            memory.append((state, agent_action, reward, next_state, done))
            
            state = next_state
            episode_rewards += reward
            
            # Process environments that completed an episode
            for i, d in enumerate(done):
                if d:
                    scores_window.append(episode_rewards[i])
                    scores.append(episode_rewards[i])
                    episode_rewards[i] = 0
                    completed_episodes += 1
                    episodes_completed_this_batch += 1

        # Learn from the batched memory
        agent.learn_from_batch(memory)
        memory.clear()
            
        if len(episode_state_vals) > 0:
            avg_state_value_history.append(np.mean(episode_state_vals))
        else:
            avg_state_value_history.append(0 if len(avg_state_value_history)==0 else avg_state_value_history[-1])

        current_avg = np.mean(scores_window) if len(scores_window) > 0 else -float('inf')
        
        for _ in range(episodes_completed_this_batch):
            lr = max(config.training.get('lr_end', 0.0), config.training.get('lr_decay', 1.0) * lr)
            agent.update_lr(lr)
        episodes_completed_this_batch = 0

        if current_avg > best_score and completed_episodes > 100:
            best_score = current_avg
            torch.save(agent.network.state_dict(), f'{output_dir}/{_run_name}_local_best.pth')

        if completed_episodes > last_logged_episode:
            if n_episodes is None:
                print(f'\rEpisode {completed_episodes}\tAverage Score: {current_avg:.2f}', end="")
                if completed_episodes - last_logged_episode >= 100 or completed_episodes % 100 == 0:
                    print(f'\rEpisode {completed_episodes}\tAverage Score: {current_avg:.2f}\tTime: {time.time() - time_ref:.0f} seconds = {(time.time() - time_ref)/60:.0f} minutes')
                    last_logged_episode = completed_episodes
            else: 
                if completed_episodes - last_logged_episode >= 10 or completed_episodes % 10 == 0:
                    print(f'\rEpisode {completed_episodes}\tAverage Score: {current_avg:.2f}\tTime: {time.time() - time_ref:.0f} seconds = {(time.time() - time_ref)/60:.0f} minutes', end="")
                    last_logged_episode = completed_episodes
                if completed_episodes % 250 == 0:
                    print(f'')

    env.close()
    torch.save(agent.network.state_dict(), f'{output_dir}/{_run_name}_last.pth')

    # Truncate scores to ensure consistent length, as vector envs might overshoot
    scores = scores[:total_episodes]

    # Interpolate state value history to match the number of episodes for consistent plotting
    if len(avg_state_value_history) > 1:
        x_orig = np.linspace(0, total_episodes, num=len(avg_state_value_history), endpoint=False)
        x_new = np.arange(total_episodes)
        interpolated_history = np.interp(x_new, x_orig, avg_state_value_history)
        avg_state_value_history = interpolated_history.tolist()
    elif avg_state_value_history: # if it has one element
        avg_state_value_history = [avg_state_value_history[0]] * total_episodes

    # For A2C, the critic's state value is the equivalent of the Q-value for logging
    # Return empty list for target Q-values to maintain consistent output signature
    return scores, [], avg_state_value_history

def reinforce(config, seed=None, record_name=None, n_episodes=None, run_type='train', study_name=None):
    """REINFORCE (Monte Carlo Policy Gradient) training loop."""
    active_env_name = config.active_env
    env_config = config.environments[active_env_name]

    # Initialize Environment
    env_params = {}
    if 'lunar_params' in env_config:
        env_params = OmegaConf.to_container(env_config.lunar_params)
    env = gym.make(active_env_name, **env_params)
    
    if not env_config.get('is_continuous', False) and isinstance(env.action_space, gym.spaces.Box):
        bins = env_config.get('discrete_bins', 3)
        env = DiscretizeBoxWrapper(env, bins)

    # --- Fake Actions Logic ---
    if env_config.get('is_continuous', False):
        real_action_size = env.action_space.shape[0]
    else:
        real_action_size = env.action_space.n
        
    agent_action_size = real_action_size
    use_fake_actions = env_config.get('use_fake_actions', False)
    if use_fake_actions:
        num_fake = env_config.get('num_fake_actions', 0)
        agent_action_size += num_fake

    agent_state_size = env.observation_space.shape[0]

    _version = config.project.version.replace(".", "-")
    _run_name = record_name if record_name is not None else config.save_parameters.run_name

    if study_name is not None:
        output_dir = f"raw_results/{active_env_name}/{_version}/{run_type}/{study_name}/{_run_name}"
    else:
        output_dir = f"raw_results/{active_env_name}/{_version}/{run_type}/{_run_name}"
    os.makedirs(output_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, "run_config.yaml"))

    current_seed = seed if seed is not None else config.project.seed
    total_episodes = n_episodes if n_episodes is not None else config.training.n_episodes

    agent = ReinforceAgent(state_size=agent_state_size, action_size=agent_action_size, config=config, seed=current_seed)
    
    lr = config.training.get('lr_start', config.agent.lr)
    agent.update_lr(lr)

    scores = []
    scores_window = deque(maxlen=100)

    best_score = -float('inf')
    time_ref = time.time()

    for i_episode in range(1, total_episodes + 1):
        state, info = env.reset(seed=current_seed*total_episodes+i_episode)
        score = 0
        memory = []
        
        for t in range(config.training.max_t):
            agent_action = agent.act(state)

            # Map agent action to real environment action
            if env_config.get('is_continuous', False):
                env_action = agent_action.copy()
            else:
                env_action = agent_action
                if use_fake_actions and agent_action >= real_action_size:
                    map_to_action = env_config.get('fake_action_maps_to', 0)
                    env_action = map_to_action

            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            # --- Sparse Reward Logic for LunarLander ---
            is_lunar_lander_sparse = (active_env_name == "LunarLander-v3" and
                                      env_config.get('sparse_reward', False))
            if is_lunar_lander_sparse and not terminated:
                reward = 0.0
            
            memory.append((state, agent_action, reward))
                
            state = next_state
            score += reward
            if done:
                break 
                
        # Learn after full episode
        agent.learn_from_episode(memory)
        
        scores_window.append(score)       
        scores.append(score)
        
        lr = max(config.training.get('lr_end', 0.0), config.training.get('lr_decay', 1.0) * lr)
        agent.update_lr(lr)

        current_avg = np.mean(scores_window)
        
        if current_avg > best_score and i_episode > 100:
            best_score = current_avg
            torch.save(agent.network.state_dict(), f'{output_dir}/{_run_name}_local_best.pth')

        if n_episodes is None:
            print(f'\rEpisode {i_episode}\tAverage Score: {current_avg:.2f}', end="")
            if i_episode % 100 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {time.time() - time_ref:.0f} seconds = {(time.time() - time_ref)/60:.0f} minutes')
                
            if i_episode % 250 == 0:
                torch.save(agent.network.state_dict(), f'{output_dir}/{_run_name}_{i_episode}-eps.pth')
                
            if current_avg >= env_config.win_condition:
                print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                torch.save(agent.network.state_dict(), f'{output_dir}/{_run_name}_best.pth')
                break
        else:
            if i_episode % 10 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {current_avg:.2f}\tTime: {time.time() - time_ref:.0f} seconds = {(time.time() - time_ref)/60:.0f} minutes', end="")
            if i_episode % 250 == 0:
                print(f'')

    if len(scores) < total_episodes:
        padding_needed = total_episodes - len(scores)
        if scores:
            scores.extend([scores[-1]] * padding_needed)
        else:
            scores.extend([0] * padding_needed)

    torch.save(agent.network.state_dict(), f'{output_dir}/{_run_name}_last.pth')    
    return scores, [], []

def modify_reward(reward, action):
    # Custom reward shaping to encourage unique landing behavior

    # add cost for main engine usage
    # if action == 2:
    #     reward -= 0.05

    # add constant penalty to encourage faster landing
    # reward -= 0.001

    return reward

if __name__ == '__main__':
    print("Starting Training!")
    
    # Construct a record name for this standalone training run
    seed = config.project.seed
    run_name = config.save_parameters.run_name
    record_name = f"{run_name}_seed{seed}"
    active_env_name = config.active_env
    version_str = config.project.version.replace('.', '-')
    output_dir = f"raw_results/{active_env_name}/{version_str}/train/{record_name}"

    # Dynamically select the training function based on the config
    algo_to_run = config.agent.get('algorithm', 'dqn')
    try:
        train_module = importlib.import_module('train')
        algo_func = getattr(train_module, algo_to_run)
    except (ImportError, AttributeError):
        print(f"Warning: could not find function '{algo_to_run}' in train.py. Falling back to dqn.")
        algo_func = dqn

    scores, q_values, avg_max_q_values = algo_func(config, record_name=record_name, run_type='train', seed=seed)
    
    # Plot the learning curve
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Plot the raw scores
    ax1.plot(np.arange(len(scores)), scores, linewidth=1.5, alpha=0.5, 
            color='#1f77b4', label='Episode Score')
    
    # Plot moving average for trend visualization
    window = 100
    moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
    ax1.plot(np.arange(window-1, len(scores)), moving_avg, linewidth=2.5, 
            color='#d62728', label='100-Episode Average')
    
    ax1.set_xlabel('Episode #', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax1.set_title('Agent Training Curve - Lunar Lander', fontsize=14, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    # Bottom Graph: Target Q-Values
    ax2.plot(np.arange(len(q_values)), q_values, color='orange')
    ax2.set_ylabel('Average Target Q-Value')
    ax2.set_title('Target Network Q-Values (from learn step)')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

    # Third Graph: Local Max Q-Values
    ax3.plot(np.arange(len(avg_max_q_values)), avg_max_q_values, color='green')
    ax3.set_ylabel('Average Max Q-Value')
    ax3.set_xlabel('Episode #')
    ax3.set_title('Local Network Max Q-Values (live state)')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{record_name}_training_curve.png", dpi=300, bbox_inches='tight')
    if config.save_parameters.get('show_plots', True):
        plt.show()
