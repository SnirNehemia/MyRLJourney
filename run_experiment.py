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

def plot_comparison(dqn_var, ddqn_var, var_name, output_path, timer_start, show_plots=True):
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

    print(f"Experiment complete! Total time: {(time.time() - timer_start)/60:.2f} minutes = {(time.time() - timer_start)/60/60:.2f} hours")

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
    plt.savefig(output_path)
    if show_plots:
        plt.show()

if __name__ == '__main__':
    config = OmegaConf.load("config.yaml")
    seeds = config.experiment.seeds
    episodes = config.experiment.n_episodes
    timer = time.time()

    version_str = config.project.version.replace('.', '-')
    experiment_name = config.save_parameters.run_name
    run_type = 'experiment'

    # This is the main directory for the experiment's summary plots
    experiment_summary_dir = f"raw_results/{version_str}/{run_type}/{experiment_name}"
    os.makedirs(experiment_summary_dir, exist_ok=True)
    
    # Dictionaries to store results
    dqn_scores, dqn_qvals = [], []
    ddqn_scores, ddqn_qvals = [], []
    
    print(" Starting Experiment: Standard DQN")
    for seed in seeds:
        print(f"Running DQN Seed {seed}...")
        record_name = f"{experiment_name}_DQN_seed{seed}"
        scores, qvals, _ = dqn(config, DQN_type="DQN", seed=seed, n_episodes=episodes, 
                               record_name=record_name, run_type=run_type)
        dqn_scores.append(scores)
        dqn_qvals.append(qvals)
        
    print("\n Starting Experiment: Double DQN")
    for seed in seeds:
        print(f"Running DDQN Seed {seed}...")
        record_name = f"{experiment_name}_DDQN_seed{seed}"
        scores, qvals, _ = dqn(config, DQN_type="DDQN", seed=seed, n_episodes=episodes,
                               record_name=record_name, run_type=run_type)
        ddqn_scores.append(scores)
        ddqn_qvals.append(qvals)

    # --- PLOTTING THE RESULTS ---
    print("\n Generating comparison graph...")
    
    show_plots = config.save_parameters.get('show_plots', True)
    plot_comparison(dqn_scores, ddqn_scores, 'Scores', 
                    output_path=f"{experiment_summary_dir}/scores_comparison.png", timer_start=timer,
                    show_plots=show_plots)
    plot_comparison(dqn_qvals, ddqn_qvals, 'Q-Values',
                    output_path=f"{experiment_summary_dir}/q_values_comparison.png", timer_start=timer,
                    show_plots=show_plots)