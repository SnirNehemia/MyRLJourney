import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
from agent import Agent, device
from omegaconf import OmegaConf
import os, shutil

config = OmegaConf.load("config.yaml")

def dqn(config, DQN_type=None, seed=None, record_name=None, n_episodes=None, run_type='train'):
    """Deep Q-Learning training loop."""

    active_env_name = config.active_env
    env_config = config.environments[active_env_name]

    # --- Fake Actions Logic ---
    real_action_size = env_config.action_size
    agent_action_size = real_action_size
    use_fake_actions = (active_env_name == "LunarLander-v3" and 
                        env_config.get('use_fake_actions', False))
    if use_fake_actions:
        num_fake = env_config.get('num_fake_actions', 6)
        agent_action_size += num_fake
        print(f"INFO: Using {num_fake} fake actions. Agent action space: {agent_action_size}")

    _version = config.project.version.replace(".", "-")
    _run_name = record_name if record_name is not None else config.save_parameters.run_name

    # Create a directory for the run
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

    # Initialize Environment and Agent
    env_params = {}
    if 'lunar_params' in env_config:
        env_params = OmegaConf.to_container(env_config.lunar_params)
    env = gym.make(active_env_name, **env_params)
    
    agent = Agent(state_size=env_config.state_size, action_size=agent_action_size, config=config, seed=current_seed)
    
    scores = []
    scores_window = deque(maxlen=100)
    q_values_history = [] 
    avg_max_q_history = []
    
    eps = config.training.eps_start
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

            agent_action = agent.act(state, eps)

            # Map agent action to real environment action
            env_action = agent_action
            if use_fake_actions and agent_action >= real_action_size:
                env_action = 0 # Map all fake actions to 'No-Op'

            next_state, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated

            # --- Sparse Reward Logic for LunarLander ---
            is_lunar_lander_sparse = (active_env_name == "LunarLander-v3" and
                                      env_config.get('sparse_reward', False))
            if is_lunar_lander_sparse and not terminated:
                reward = 0.0
            
            # The agent.step call needs the original agent_action to learn correctly
            q_val = agent.step(state, agent_action, reward, next_state, done, tau)
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

        eps = max(config.training.eps_end, config.training.eps_decay * eps) 
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
                print(f'\rEpisode {i_episode}\tAverage Score: {current_avg:.2f}', end="")
            if i_episode % 250 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {time.time() - time_ref:.0f} seconds = {(time.time() - time_ref)/60:.0f} minutes')

    torch.save(agent.qnetwork_local.state_dict(), f'{output_dir}/{_run_name}_last.pth')    
    return scores, q_values_history, avg_max_q_history

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

    scores, q_values, avg_max_q_values = dqn(config, record_name=record_name, run_type='train', seed=seed)
    
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
