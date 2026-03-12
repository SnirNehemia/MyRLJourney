import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import os
import time

from train import dqn

def plot_ablation_statistics(results_dict, title, y_label, output_path, win_condition=None, show_plots=True):
    """
    Plots the mean and standard deviation of multiple runs for different configurations.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    colors = [
        '#1f77b4',  # Muted Blue
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

    active_env_name = base_config.active_env
    env_config = base_config.environments[active_env_name]

    n_episodes = base_config.ablation_study.get('n_episodes', 1000)
    study_name = base_config.ablation_study.get('ablation_name', 'ablation_study')
    seeds = base_config.ablation_study.get('seeds', [0])
    version_str = base_config.project.version.replace('.', '-')
    run_type = 'ablation'

    # This will be the main folder for the study's output plot
    study_summary_dir = f"raw_results/{active_env_name}/{version_str}/{run_type}/{study_name}"
    os.makedirs(study_summary_dir, exist_ok=True)

    # --- Define configurations to test ---
    configs_to_test = []
    study_type = base_config.ablation_study.get('study_type', 'component')

    if study_type == 'sweep':
        print("--- Running Parameter Sweep Study ---")
        sweep_config = base_config.ablation_study.sweep
        param_to_sweep = sweep_config.parameter
        sweep_values = sweep_config['list_values']
        
        param_name_for_label = param_to_sweep.split('.')[-1]
        for value in sweep_values:
            configs_to_test.append({
                'name': f"{param_name_for_label}={value}",
                'study_type': 'sweep',
                'param': param_to_sweep,
                'value': value
            })
    elif study_type == 'dqn_variants':
        print("--- Running DQN Variants Study ---")
        configs_to_test = [
            {'name': 'DQN (No Target)', 'study_type': 'dqn_variants', 'target': False, 'dqn_type': 'DQN'},
            {'name': 'DQN (With Target)', 'study_type': 'dqn_variants', 'target': True, 'dqn_type': 'DQN'},
            {'name': 'Double DQN', 'study_type': 'dqn_variants', 'target': True, 'dqn_type': 'DDQN'},
        ]
    else: # 'component'
        print("--- Running Component Ablation Study ---")
        configs_to_test = [
            {'name': 'Full DQN (Buffer, Target)', 'study_type': 'component', 'buffer': True, 'target': True},
            {'name': 'No Replay Buffer', 'study_type': 'component', 'buffer': False, 'target': True},
            {'name': 'No Target Network', 'study_type': 'component', 'buffer': True, 'target': False},
            {'name': 'Naive DQN (No Buffer, No Target)', 'study_type': 'component', 'buffer': False, 'target': False},
        ]

    results = {}
    q_results = {}
    timer = time.time()

    for cfg_mod in configs_to_test:
        print(f"\n--- Running Ablation: {cfg_mod['name']} ---")
        
        all_seed_scores = []
        all_seed_q_vals = []

        for seed in seeds:
            print(f"  Running seed: {seed}")
            run_config = base_config.copy()
            
            # Apply modifications for the current run
            current_study_type = cfg_mod['study_type']
            if current_study_type == 'sweep':
                OmegaConf.update(run_config, cfg_mod['param'], cfg_mod['value'])
            elif current_study_type == 'component':
                run_config.agent.use_replay_buffer = cfg_mod['buffer']
                run_config.agent.use_target_network = cfg_mod['target']
            elif current_study_type == 'dqn_variants':
                run_config.agent.DQN_type = cfg_mod['dqn_type']
                run_config.agent.use_target_network = cfg_mod['target']
            
            # Define a unique name for this run's artifacts and folder
            record_name = f"{study_name}_{cfg_mod['name'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '_')}_seed{seed}"
            
            # Run the training. The dqn function will create a unique folder for this run.
            scores, _, avg_max_q = dqn(
                config=run_config, 
                n_episodes=n_episodes, 
                record_name=record_name, 
                run_type=run_type,
                seed=seed
            )
            
            all_seed_scores.append(scores)
            all_seed_q_vals.append(avg_max_q)

        results[cfg_mod['name']] = all_seed_scores
        q_results[cfg_mod['name']] = all_seed_q_vals
        print(f"\n--- Finished: {cfg_mod['name']} ---")

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