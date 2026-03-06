import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import os
import time

from train import dqn

if __name__ == '__main__':
    base_config = OmegaConf.load("config.yaml")
    n_episodes = base_config.ablation_study.get('n_episodes', 2500)
    run_name_prefix = "ablation_study"

    # This will be the main folder for the study's output plot
    study_output_dir = f"raw_results/{base_config.project.version.replace('.', '-')}_{run_name_prefix}"
    os.makedirs(study_output_dir, exist_ok=True)

    configs_to_test = [
        {'name': 'Full DQN (Buffer, Target)', 'buffer': True, 'target': True},
        {'name': 'No Replay Buffer', 'buffer': False, 'target': True},
        {'name': 'No Target Network', 'buffer': True, 'target': False},
        {'name': 'Naive DQN (No Buffer, No Target)', 'buffer': False, 'target': False},
    ]

    results = {}
    q_results = {}
    timer = time.time()

    for cfg_mod in configs_to_test:
        print(f"\n--- Running Ablation: {cfg_mod['name']} ---")
        # Create a copy of the base config to modify for the run
        run_config = base_config.copy()
        
        # Apply modifications for the current ablation run
        run_config.agent.use_replay_buffer = cfg_mod['buffer']
        run_config.agent.use_target_network = cfg_mod['target']
        run_config.project.seed = run_config.ablation_study.seed
        # Define a unique name for this run's artifacts and folder
        record_name = f"{run_name_prefix}_{cfg_mod['name'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')}"
        
        # Run the training. The dqn function will create a unique folder for this run.
        scores, _, avg_max_q = dqn(
            config=run_config, 
            n_episodes=n_episodes, 
            record_name=record_name
        )
        
        results[cfg_mod['name']] = scores
        q_results[cfg_mod['name']] = avg_max_q
        print(f"\n--- Finished: {cfg_mod['name']} ---")

    print(f"\nAblation study finished in {(time.time() - timer)/60:.2f} minutes.")

    # --- Plotting the comparison ---
    print("Generating comparison plot...")
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, layout="constrained")
    fig.suptitle('DQN Ablation Study: Lunar Lander Performance', fontsize=22, fontweight='bold')

    axs = axs.flatten()
    window = 100

    for i, (name, scores) in enumerate(results.items()):
        scores = np.array(scores)
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        
        # Plotting
        axs[i].plot(scores, color='skyblue', alpha=0.5, label='Episode Score', linewidth=1)
        axs[i].plot(np.arange(window-1, len(scores)), moving_avg, color='#003366', linewidth=2.5, label=f'{window}-Ep Avg.')
        axs[i].axhline(y=200, color='#d62728', linestyle='--', linewidth=1.5, label='Win Condition')
        
        # Text Annotation for Success Rate
        # Calculate Success Rate (%)
        success_rate = (np.sum(scores >= run_config.training.win_condition) / len(scores)) * 100
        # Transform=axs[i].transAxes allows us to place text relative to the plot (0,0 is bottom left, 1,1 is top right)
        axs[i].text(0.05, 0.92, f'Success Rate: {success_rate:.1f}%', 
        transform=axs[i].transAxes, fontsize=12, fontweight='bold', 
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Clean titles
        axs[i].set_title(name, fontsize=16, fontweight='semibold', pad=10)
        axs[i].grid(True, linestyle=':', alpha=0.6)
        axs[i].set_ylim(-400, 275)

    fig.supxlabel('Episode #', fontsize=14, fontweight='bold')
    fig.supylabel('Score', fontsize=14, fontweight='bold')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=12)

    plt.savefig(f"{study_output_dir}/ablation_comparison.png", dpi=300)
    plt.show()

    # --- Plotting the Q-Value comparison ---
    print("Generating Q-value comparison plot...")
    plt.style.use('seaborn-v0_8-whitegrid') 
    fig_q, axs_q = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, layout="constrained")
    fig_q.suptitle('Ablation Study: Average Max Q-Value (Local Network)', fontsize=22, fontweight='bold')

    axs_q = axs_q.flatten()

    for i, (name, q_values) in enumerate(q_results.items()):
        q_values = np.array(q_values)
        
        # Plotting
        axs_q[i].plot(q_values, color='#2ca02c', linewidth=2, label='Avg. Max Q-Value')
        
        # Clean titles
        axs_q[i].set_title(name, fontsize=16, fontweight='semibold', pad=10)
        axs_q[i].grid(True, linestyle=':', alpha=0.6)

    fig_q.supxlabel('Episode #', fontsize=14, fontweight='bold')
    fig_q.supylabel('Average Max Q-Value per Episode', fontsize=14, fontweight='bold')

    handles, labels = axs_q[0].get_legend_handles_labels()
    fig_q.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), fontsize=12)

    plt.savefig(f"{study_output_dir}/ablation_q_values_comparison.png", dpi=300)
    plt.show()