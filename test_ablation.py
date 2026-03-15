import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from omegaconf import OmegaConf

from agent import Agent, device

def test_network(model_path, config, n_episodes=100):
    """
    Tests a trained agent network over a fixed number of episodes with deterministic seeds.

    Args:
        model_path (str): Path to the saved .pth model file.
        config (OmegaConf): The configuration object for the run.
        n_episodes (int): Number of episodes to test.

    Returns:
        tuple: A tuple containing (list of scores, list of avg_max_q_values).
    """
    if not os.path.exists(model_path):
        print(f"  Warning: Model file not found at {model_path}. Skipping test.")
        return None, None

    print(f"  Testing model: {os.path.basename(model_path)}")

    # Initialize environment and agent
    active_env_name = config.active_env
    env_config = config.environments[active_env_name]
    env_params = {}
    if 'lunar_params' in env_config:
        env_params = OmegaConf.to_container(env_config.lunar_params)
    env = gym.make(active_env_name, **env_params)
    
    # --- Fake Actions Logic ---
    real_action_size = env_config.action_size
    agent_action_size = real_action_size
    use_fake_actions = env_config.get('use_fake_actions', False)
    if use_fake_actions:
        num_fake = env_config.get('num_fake_actions', 0)
        agent_action_size += num_fake

    agent = Agent(state_size=env_config.state_size, action_size=agent_action_size, config=config, seed=0)
    agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    agent.qnetwork_local.eval() # Set model to evaluation mode

    scores = []
    avg_max_q_history = []

    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=i_episode) # Use a different, fixed seed for each test episode
        score = 0
        episode_max_q_vals = []
        
        for t in range(config.training.max_t):
            # Get max predicted Q-value for the current state
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_values = agent.qnetwork_local(state_tensor)
            episode_max_q_vals.append(torch.max(action_values).item())

            # Act greedily
            agent_action = agent.act(state, eps=0.0)

            # Map agent action to real environment action
            env_action = agent_action
            if use_fake_actions and agent_action >= real_action_size:
                map_to_action = env_config.get('fake_action_maps_to', 0)
                env_action = map_to_action

            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated
            
            state = next_state
            score += reward
            if done:
                break
        
        scores.append(score)
        if episode_max_q_vals:
            avg_max_q_history.append(np.mean(episode_max_q_vals))
        else:
            avg_max_q_history.append(0)

    env.close()
    return scores, avg_max_q_history

def plot_violin(data_dict, title, y_label, output_path, show_plots=True):
    """
    Creates and saves a violin plot for the given data.
    """
    labels = list(data_dict.keys())
    data = list(data_dict.values())

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    violin_parts = ax.violinplot(data, showmeans=False, showmedians=True)
    
    for pc in violin_parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)

    violin_parts['cmedians'].set_edgecolor('red')
    violin_parts['cmedians'].set_linewidth(2)

    for partname in ('cbars', 'cmins', 'cmaxes'):
        vp = violin_parts[partname]
        vp.set_edgecolor('none')
        vp.set_linewidth(0)

    ax.yaxis.grid(True)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    if show_plots:
        plt.show()

def plot_scatter(scores_dict, q_values_dict, title, output_path, show_plots=True):
    """
    Creates and saves a grid of scatter plots of score vs. avg max q-value.
    """
    labels = list(scores_dict.keys())
    num_plots = len(labels)
    
    # Determine grid size
    if num_plots <= 2:
        nrows, ncols = 1, num_plots
        figsize = (8 * ncols, 6)
    elif num_plots <= 4:
        nrows, ncols = 2, 2
        figsize = (14, 10)
    else: # a bit more general for sweeps
        ncols = 3
        nrows = (num_plots + ncols - 1) // ncols
        figsize = (18, 6 * nrows)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    if num_plots > 1:
        axs = np.array(axs).flatten()
    else:
        axs = [axs] # make it iterable

    fig.suptitle(title, fontsize=20, fontweight='bold')

    for i, name in enumerate(labels):
        ax = axs[i]
        scores = np.array(scores_dict[name])
        q_values = np.array(q_values_dict[name])
        
        # Color points by score for better visualization
        scatter = ax.scatter(q_values, scores, alpha=0.6, edgecolors='k', s=35, c=scores, cmap='viridis')
        
        # Calculate and display correlation
        correlation = np.corrcoef(q_values, scores)[0, 1]
        ax.text(0.05, 0.95, f'Corr: {correlation:.2f}', transform=ax.transAxes,
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
        
        ax.set_title(name, fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)

    # Add common labels
    fig.supxlabel('Average Max Q-Value per Episode', fontsize=14)
    fig.supylabel('Final Episode Score', fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    plt.savefig(output_path, dpi=300)
    if show_plots:
        plt.show()

def test_ablation():
    """
    Runs a full test suite on the results of an ablation study, generating a
    statistical report and violin plots for visual comparison.
    """
    base_config = OmegaConf.load("config.yaml")
    active_env_name = base_config.active_env
    study_name = base_config.ablation_study.get('ablation_name', 'ablation_study')
    first_seed = base_config.ablation_study.get('seeds', [0])[0]
    version_str = base_config.project.version.replace('.', '-')
    run_type = 'ablation'

    study_summary_dir = f"raw_results/{active_env_name}/{version_str}/{run_type}/{study_name}"
    
    configs_to_test = []
    study_type = base_config.ablation_study.get('study_type', 'component')

    if study_type == 'sweep':
        print("--- Locating Models for Parameter Sweep Study ---")
        sweep_config = base_config.ablation_study.sweep
        param_to_sweep = sweep_config.parameter
        sweep_values = sweep_config['list_values']
        param_name_for_label = param_to_sweep.split('.')[-1]
        for value in sweep_values:
            configs_to_test.append({'name': f"{param_name_for_label}={value}"})
    elif study_type == 'dqn_variants':
        print("--- Locating Models for DQN Variants Study ---")
        configs_to_test = [
            {'name': 'DQN (No Target)'},
            {'name': 'DQN (With Target)'},
            {'name': 'Double DQN'},
        ]
    else: # 'component'
        print("--- Locating Models for Component Ablation Study ---")
        configs_to_test = [
            {'name': 'Full DQN (Buffer, Target)'}, {'name': 'No Replay Buffer'},
            {'name': 'No Target Network'}, {'name': 'Naive DQN (No Buffer, No Target)'}
        ]

    all_test_scores, all_test_q_values = {}, {}

    for cfg_mod in configs_to_test:
        print(f"\nProcessing configuration: {cfg_mod['name']}")
        run_name_slug = cfg_mod['name'].replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '_')
        record_name = f"{study_name}_{run_name_slug}_seed{first_seed}"
        model_dir = f"raw_results/{active_env_name}/{version_str}/{run_type}/{record_name}"
        model_path = os.path.join(model_dir, f"{record_name}_local_best.pth")
        run_config_path = os.path.join(model_dir, "run_config.yaml")

        if not os.path.exists(run_config_path):
            print(f"  Warning: Run config not found at {run_config_path}. Skipping.")
            continue
        
        run_config = OmegaConf.load(run_config_path)
        scores, q_values = test_network(model_path, run_config)
        
        if scores is not None:
            all_test_scores[cfg_mod['name']] = scores
            all_test_q_values[cfg_mod['name']] = q_values

    print("\n\n--- Ablation Test Suite Report ---")
    for name in all_test_scores.keys():
        scores, q_values = np.array(all_test_scores[name]), np.array(all_test_q_values[name])
        print(f"\n--- Configuration: {name} ---")
        print(f"  Scores:    Avg: {np.mean(scores):.2f}, Std: {np.std(scores):.2f}, Median: {np.median(scores):.2f}, Min: {np.min(scores):.2f}, Max: {np.max(scores):.2f}")
        print(f"  Q-Values:  Avg: {np.mean(q_values):.2f}, Std: {np.std(q_values):.2f}, Median: {np.median(q_values):.2f}, Min: {np.min(q_values):.2f}, Max: {np.max(q_values):.2f}")
    print("\n--- End of Report ---\n")

    if all_test_scores:
        print("Generating comparison plots...")
        show_plots = base_config.save_parameters.get('show_plots', True)
        plot_violin(all_test_scores, 'Test Score Distribution Across Ablation Configurations',
                    'Score (100 test episodes per configuration)',
                    os.path.join(study_summary_dir, 'test_scores_violin.png'),
                    show_plots=show_plots)
        plot_violin(all_test_q_values, 'Test Avg. Max Q-Value Distribution Across Configurations',
                    'Average Max Q-Value per Episode',
                    os.path.join(study_summary_dir, 'test_q_values_violin.png'),
                    show_plots=show_plots)
        plot_scatter(all_test_scores, all_test_q_values,
                     'Test Score vs. Avg Max Q-Value Correlation',
                     os.path.join(study_summary_dir, 'test_score_vs_q_scatter.png'),
                     show_plots=show_plots)

if __name__ == '__main__':
    test_ablation()