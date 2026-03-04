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
# 1. Initialize the Environment and Agent
env = gym.make('LunarLander-v3',
               enable_wind=True, 
               wind_power=config.lunar_params.wind, 
               turbulence_power=config.lunar_params.turbulence,
               gravity=config.lunar_params.gravity)
# State size is 8 (coordinates, velocity, angle, etc.)
# Action size is 4 (engines: none, left, main, right)
agent = Agent(state_size=8, action_size=4, seed=config.project.seed)

def dqn(cfg=config, n_episodes=None, max_t=None,
         eps_start=None, eps_end=None, eps_decay=None,
         win_condition=None):
    """Deep Q-Learning loop."""
    # Resolve parameters using config defaults when not provided
    training = cfg.training
    n_episodes = training.n_episodes if n_episodes is None else n_episodes
    max_t = training.max_t if max_t is None else max_t
    eps_start = training.eps_start if eps_start is None else eps_start
    eps_end = training.eps_end if eps_end is None else eps_end
    eps_decay = training.eps_decay if eps_decay is None else eps_decay
    win_condition = training.win_condition if win_condition is None else win_condition

    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores for our win condition
    eps = eps_start                    # initialize epsilon
    time_ref = time.time()
    best_score = -float('inf')
    q_values_history = []

    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset()
        score = 0
        episode_q_vals = []

        for t in range(max_t):
            # 1. The agent picks an action
            action = agent.act(state, eps)
            
            # 2. The environment reacts
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            reward = modify_reward(reward, action)

            # 3. The agent learns from the result
            q_val = agent.step(state, action, reward, next_state, done)
            if q_val is not None:
                episode_q_vals.append(q_val)

            state = next_state
            score += reward
            if done:
                break 
                
        if len(episode_q_vals) > 0:
            q_values_history.append(np.mean(episode_q_vals))
        else:
            # If no learning happened (e.g., very first few frames), append 0
            q_values_history.append(0 if len(q_values_history)==0 else q_values_history[-1])
        # Save scores
        scores_window.append(score)       
        scores.append(score)              
        
        # Decrease epsilon (less random exploration over time)
        eps = max(eps_end, eps_decay * eps) 
        
        # Print progress to the terminal
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.0f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {time.time() - time_ref:.0f} seconds = {(time.time() - time_ref)/60:.0f} minutes')
            
        # Save whenever we hit a new high score
        current_avg = np.mean(scores_window)
        if current_avg > best_score and i_episode > 1000:
            best_score = current_avg
            torch.save(agent.qnetwork_local.state_dict(),
                        f'raw_results/{full_run_name}_local_best.pth')

        if i_episode % 250 == 0:
            torch.save(agent.qnetwork_local.state_dict(),
                        f'raw_results/{full_run_name}_{i_episode}-eps.pth')

        # Check Win Condition
        if current_avg >= win_condition:
            print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            # Save the trained neural network weights!
            torch.save(agent.qnetwork_local.state_dict(), f'raw_results/{full_run_name}_best.pth')
            break
            
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
    scores, q_values = dqn()

    # Plot the learning curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), dpi=100)
    ax = fig.add_subplot(111)
    
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
