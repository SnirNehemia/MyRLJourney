import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import os
import time
import importlib

from train import dqn # Fallback logic

def plot_ablation_statistics(results_dict, title, y_label, output_path, win_condition=None, show_plots=True):
    """
    Plots the mean and standard deviation of multiple runs for different configurations.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    colors = [
        "#9421ce",  # Muted Blue
        '#ff7f0e',  # Safety Orange
        '#2ca02c',  # Cooked Asparagus Green
        '#d62728',  # Brick Red
        '#9467bd',  # Muted Purple
        '#8c564b',  # Chestnut Brown
        '#e377c2',  # Raspberry Pink
        '#7f7f7f',  # Middle Gray
    ]
    
    for i, (name, runs) in enumerate(results_dict.items()):
        runs_np = np.array(runs)
        
        mean = np.mean(runs_np, axis=0)
        std = np.std(runs_np, axis=0)
        
        window = 100
        def moving_average(data, window_size=100):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            
        mean_smooth = moving_average(mean, window)
        std_smooth = moving_average(std, window)
        x_axis = np.arange(len(mean_smooth))
        
        plt.plot(x_axis, mean_smooth, label=name, color=colors[i % len(colors)], linewidth=2.5)
        plt.fill_between(x_axis, mean_smooth - std_smooth, mean_smooth + std_smooth, 
                         color=colors[i % len(colors)], alpha=0.15)

    if win_condition is not None:
        plt.axhline(y=win_condition, color='gray', linestyle='--', label=f'Win Condition ({win_condition})')
        
    plt.title(title, fontsize=18, fontweight='bold')
    plt.xlabel('Episode # (smoothed over 100 episodes)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if show_plots:
        plt.show()

def run_ablation_study():
    base_config = OmegaConf.load("config.yaml")
    ablation_config = base_config.ablation_study

    active_env_name = base_config.active_env
    env_config = base_config.environments[active_env_name]

    n_episodes = ablation_config.get('n_episodes', 1000)
    study_name = ablation_config.get('study_name', 'ablation_study')
    seeds = ablation_config.get('seeds', [0])
    version_str = base_config.project.version.replace('.', '-')
    run_type = 'ablation'

    # This will be the main folder for the study's output plot
    study_summary_dir = f"raw_results/{active_env_name}/{version_str}/{run_type}/{study_name}"
    os.makedirs(study_summary_dir, exist_ok=True)

    results = {}
    q_results = {}
    timer = time.time()

    for experiment in ablation_config.experiments:
        exp_name = experiment['name']
        print(f"\n--- Running Ablation: {exp_name} ---")
        
        all_seed_scores = []
        all_seed_q_vals = []

        for seed in seeds:
            print(f"  Running seed: {seed}")
            run_config = base_config.copy()
            
            # Create a dictionary of overrides, excluding the 'name' key
            overrides = {k: v for k, v in experiment.items() if k != 'name'}
            
            # Merge the overrides into the run configuration
            OmegaConf.set_struct(run_config, False) # Allow adding new keys
            run_config = OmegaConf.merge(run_config, overrides)
            OmegaConf.set_struct(run_config, True)
            
            # Define a unique name for this run's artifacts and folder
            record_name = f"{study_name}_{exp_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '_')}_seed{seed}"
            
            algo_to_run = run_config.agent.get('algorithm', 'dqn')
            try:
                train_module = importlib.import_module('train')
                algo_func = getattr(train_module, algo_to_run)
            except (ImportError, AttributeError):
                print(f"Warning: could not find function '{algo_to_run}' in train.py. Falling back to dqn.")
                algo_func = dqn

            # Run the training. The algorithm function will create a unique folder for this run.
            scores, _, avg_max_q = algo_func(
                config=run_config, 
                n_episodes=n_episodes, 
                record_name=record_name, 
                run_type=run_type,
                study_name=study_name,
                seed=seed
            )
            
            all_seed_scores.append(scores)
            all_seed_q_vals.append(avg_max_q)

        results[exp_name] = all_seed_scores
        q_results[exp_name] = all_seed_q_vals
        print(f"\n--- Finished: {exp_name} ---")

    print(f"\nAblation study finished in {(time.time() - timer)/60:.2f} minutes.")

    # --- Plotting the comparison ---
    print("\nGenerating comparison plots...")
    
    show_plots = base_config.save_parameters.get('show_plots', True)
    plot_ablation_statistics(
        results_dict=results,
        title='Ablation Study: Agent Performance (Mean & Std Dev)',
        y_label='Average Score',
        output_path=f"{study_summary_dir}/scores_comparison.png",
        win_condition=env_config.win_condition,
        show_plots=show_plots
    )

    plot_ablation_statistics(
        results_dict=q_results,
        title='Ablation Study: Average Max Q-Value (Mean & Std Dev)',
        y_label='Average Max Q-Value',
        output_path=f"{study_summary_dir}/q_values_comparison.png",
        show_plots=show_plots
    )

if __name__ == '__main__':
    run_ablation_study()