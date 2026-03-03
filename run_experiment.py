import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent import Agent
from omegaconf import OmegaConf
import time 

config = OmegaConf.load("config.yaml")

__version__ = config.project.version

def train_for_experiment(DQN_type, n_episodes=config.experiment.n_episodes, seed=config.project.seed):
    """Modified training loop that runs for a fixed number of episodes."""
    env = gym.make('LunarLander-v3')
    agent = Agent(state_size=8, action_size=4, seed=seed, DQN_type=DQN_type)
    
    scores = []
    scores_window = deque(maxlen=100)
    eps = 1.0
    eps_decay = 0.995
    eps_end = 0.01
    
    time_ref = time.time()

    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset(seed=seed) # Set the seed for the environment!
        score = 0
        
        for t in range(config.training.max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
                
        scores_window.append(score)       
        scores.append(score)              
        eps = max(eps_end, eps_decay * eps) 
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window[-1]):.2f}\tTime: {time.time() - time_ref:.2f} seconds = {(time.time() - time_ref)/60:.2f} minutes')
    print(f"Training complete for {DQN_type} with seed {seed}! Total time: {time.time() - time_ref:.2f} seconds = {(time.time() - time_ref)/60:.2f} minutes")
    return scores

if __name__ == '__main__':
    seeds = config.experiment.seeds
    episodes = config.experiment.n_episodes
    timer = time.time()
    
    # Dictionaries to store results
    dqn_results = []
    ddqn_results = []
    
    print(" Starting Experiment: Standard DQN")
    for seed in seeds:
        print(f"Running DQN Seed {seed}...")
        scores = train_for_experiment(DQN_type="DQN", seed=seed, n_episodes=episodes)
        dqn_results.append(scores)
        
    print("\n Starting Experiment: Double DQN")
    for seed in seeds:
        print(f"Running DDQN Seed {seed}...")
        scores = train_for_experiment(DQN_type="DDQN", seed=seed, n_episodes=episodes)
        ddqn_results.append(scores)

    # --- PLOTTING THE RESULTS ---
    print("\n Generating comparison graph...")
    
    # Convert lists to numpy arrays for easy math: shape will be (3, 800)
    dqn_results = np.array(dqn_results)
    ddqn_results = np.array(ddqn_results)
    
    # Calculate Mean and Standard Deviation across the 3 seeds
    dqn_mean = np.mean(dqn_results, axis=0)
    dqn_std = np.std(dqn_results, axis=0)
    
    ddqn_mean = np.mean(ddqn_results, axis=0)
    ddqn_std = np.std(ddqn_results, axis=0)
    
    # Calculate a moving average to smooth the noisy lines (window of 50)
    def moving_average(data, window=50):
        return np.convolve(data, np.ones(window)/window, mode='valid')
        
    dqn_mean_smooth = moving_average(dqn_mean)
    ddqn_mean_smooth = moving_average(ddqn_mean)
    x_axis = np.arange(len(dqn_mean_smooth))

    print(f"Experiment complete! Total time: {(time.time() - timer)/60:.2f} minutes = {(time.time() - timer)/60/60:.2f} hours")

    plt.figure(figsize=(10, 6))
    
    # Plot standard DQN (Blue)
    plt.plot(x_axis, dqn_mean_smooth, label='Standard DQN', color='blue')
    plt.fill_between(x_axis, 
                     dqn_mean_smooth - moving_average(dqn_std), 
                     dqn_mean_smooth + moving_average(dqn_std), 
                     color='blue', alpha=0.2)

    # Plot Double DQN (Orange)
    plt.plot(x_axis, ddqn_mean_smooth, label='Double DQN', color='darkorange')
    plt.fill_between(x_axis, 
                     ddqn_mean_smooth - moving_average(ddqn_std), 
                     ddqn_mean_smooth + moving_average(ddqn_std), 
                     color='darkorange', alpha=0.2)

    plt.axhline(y=200, color='r', linestyle='--', label='Win Condition (200)')
    plt.title('DQN vs Double DQN: Lunar Lander Performance')
    plt.xlabel('Episode #')
    plt.ylabel('Average Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save it so you can put it in your README!
    plt.savefig("results/dqn_vs_ddqn_comparison.png")
    plt.show()
