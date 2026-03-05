import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from agent import Agent
from omegaconf import OmegaConf
import time 
import os

from train import dqn

config = OmegaConf.load("config.yaml")
__version__ = config.project.version
full_run_name = __version__.replace(".", "-") + "_"+ config.save_parameters.run_name
output_dir = f"raw_results/{full_run_name}"

def plot_comparison(dqn_var, ddqn_var, var_name):
    # Convert lists to numpy arrays for easy math: shape will be (3, 800)
    dqn_results = np.array(dqn_var)
    ddqn_results = np.array(ddqn_var)
    
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
    plt.title(f'DQN vs Double DQN: Lunar Lander Performance: {var_name}')
    plt.xlabel('Episode #')
    plt.ylabel(f'Average {var_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save it so you can put it in your README!
    plt.savefig(f"{output_dir}/{full_run_name}_{var_name}_comparison.png")
    plt.show()

if __name__ == '__main__':
    seeds = config.experiment.seeds
    episodes = config.experiment.n_episodes
    timer = time.time()
    
    # Dictionaries to store results
    dqn_scores, dqn_qvals = [], []
    ddqn_scores, ddqn_qvals = [], []
    
    print(" Starting Experiment: Standard DQN")
    for seed in seeds:
        print(f"Running DQN Seed {seed}...")
        scores, qvals = dqn(config, DQN_type="DQN", seed=seed,
                            n_episodes=config.experiment.n_episodes, record_name=f"{full_run_name}_DQN_{seed}")
        dqn_scores.append(scores)
        dqn_qvals.append(qvals)
        
    print("\n Starting Experiment: Double DQN")
    for seed in seeds:
        print(f"Running DDQN Seed {seed}...")
        scores, qvals = dqn(config, DQN_type="DDQN", seed=seed,
                            n_episodes=config.experiment.n_episodes, record_name=f"{full_run_name}_DDQN_{seed}")
        ddqn_scores.append(scores)
        ddqn_qvals.append(qvals)

    # --- PLOTTING THE RESULTS ---
    print("\n Generating comparison graph...")
    
    plot_comparison(dqn_scores, ddqn_scores, 'Scores')
    plot_comparison(dqn_qvals, ddqn_qvals, 'Q-Values')