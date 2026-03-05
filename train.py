import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
from agent import Agent
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
__version__ = config.project.version

full_run_name = __version__.replace(".", "-") + "_"+ config.save_parameters.run_name

def dqn(config, DQN_type=None, seed=None, record_name=None, n_episodes=None):
    """Deep Q-Learning training loop."""
    
    # Use config defaults unless overridden by experiment arguments
    DQN_type = DQN_type if DQN_type is not None else config.agent.DQN_type
    print(f"DEBUG: dqn running with DQN_type={DQN_type}")
    current_seed = seed if seed is not None else config.project.seed
    total_episodes = n_episodes if n_episodes is not None else config.training.n_episodes

    # Initialize Environment and Agent
    env = gym.make('LunarLander-v3',
               enable_wind=True, 
               wind_power=config.lunar_params.wind, 
               turbulence_power=config.lunar_params.turbulence,
               gravity=config.lunar_params.gravity)
    
    agent = Agent(state_size=8, action_size=4, seed=current_seed, DQN_type=DQN_type)
    
    scores = []
    scores_window = deque(maxlen=100)
    q_values_history = [] 
    
    eps = config.training.eps_start
    best_score = -float('inf')
    time_ref = time.time()

    for i_episode in range(1, total_episodes + 1):
        state, info = env.reset(seed=current_seed)
        score = 0
        episode_q_vals = []
        
        for t in range(config.training.max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            q_val = agent.step(state, action, reward, next_state, done)
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
        # # Print progress to the terminal
        # if i_episode % 10 == 0:
        #     print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.0f}', end="")
        # if i_episode % 100 == 0:
        #     print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {time.time() - time_ref:.0f} seconds = {(time.time() - time_ref)/60:.0f} minutes')
        eps = max(config.training.eps_end, config.training.eps_decay * eps) 
        current_avg = np.mean(scores_window)
        
        # Only print progress if we are running a standard training session (not an experiment)
        if n_episodes is None:
            print(f'\rEpisode {i_episode}\tAverage Score: {current_avg:.2f}', end="")
            if i_episode % 100 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {time.time() - time_ref:.0f} seconds = {(time.time() - time_ref)/60:.0f} minutes')
                
            # Save best model logic (only during normal training)
            if current_avg > best_score and i_episode > 100:
                best_score = current_avg
                torch.save(agent.qnetwork_local.state_dict(),
                        f'raw_results/{record_name}_local_best.pth')
                
            if i_episode % 250 == 0:
                torch.save(agent.qnetwork_local.state_dict(),
                            f'raw_results/{record_name}_{i_episode}-eps.pth')
                
            # Check Win Condition
            if current_avg >= config.training.win_condition:
                print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
                # Save the trained neural network weights!
                torch.save(agent.qnetwork_local.state_dict(), f'raw_results/{record_name}_best.pth')
                break
        else: 
            if i_episode % 10 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {current_avg:.2f}', end="")
            if i_episode % 250 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {time.time() - time_ref:.0f} seconds = {(time.time() - time_ref)/60:.0f} minutes')

    torch.save(agent.qnetwork_local.state_dict(), f'raw_results/{record_name}_last.pth')    
    return scores, q_values_history

def modify_reward(reward, action):
    # Custom reward shaping to encourage unique landing behavior
    # Discourage the agent from using the main engine (action 2), out of interest of seeing more creative solutions.
    # This is optional and can be adjusted based on your preferences.
    # if action == 2:
    #     reward -= 0.05

    return reward

if __name__ == '__main__':
    print("Starting Training!")
    # we can still call without arguments since cfg defaults to the global
    # configuration, but tests or other scripts may pass a different cfg.
    scores, q_values = dqn(config, record_name=full_run_name)

    # Plot the learning curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    
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
    ax2.set_xlabel('Episode #')
    ax2.set_title('Network Confidence (Target Q-Values)')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)

    plt.tight_layout()
    plt.savefig(f"raw_results/{full_run_name}_training_curve.png", dpi=300, bbox_inches='tight')
    plt.show()
