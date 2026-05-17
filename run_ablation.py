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
        
        # Skip plotting if the algorithm didn't return any data (e.g., REINFORCE has no Q-values)
        if runs_np.size == 0:
            continue
            
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
    grad_step_results = {}
    ev_results = {}
    actor_loss_results = {}
    critic_loss_results = {}
    entropy_loss_results = {}
    raw_entropy_results = {}
    actor_grad_results = {}
    critic_grad_results = {}
    ep_len_results = {}
    timer = time.time()

    for experiment in ablation_config.experiments:
        exp_name = experiment['name']
        print(f"\n--- Running Ablation: {exp_name} ---")

        all_seed_scores = []
        all_seed_q_vals = []
        all_seed_grad_scores = []
        all_seed_ev = []
        all_seed_actor_loss = []
        all_seed_critic_loss = []
        all_seed_entropy_loss = []
        all_seed_raw_entropy = []
        all_seed_actor_grad = []
        all_seed_critic_grad = []
        all_seed_ep_len = []

        for seed in seeds:
            print(f"  Running seed: {seed}")
            run_config = base_config.copy()

            overrides = {k: v for k, v in experiment.items() if k != 'name'}
            OmegaConf.set_struct(run_config, False)
            run_config = OmegaConf.merge(run_config, overrides)
            OmegaConf.set_struct(run_config, True)

            record_name = f"{study_name}_{exp_name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').replace('=', '_')}_seed{seed}"

            algo_to_run = run_config.agent.get('algorithm', 'dqn')
            try:
                train_module = importlib.import_module('train')
                algo_func = getattr(train_module, algo_to_run)
            except (ImportError, AttributeError):
                print(f"Warning: could not find function '{algo_to_run}' in train.py. Falling back to dqn.")
                algo_func = dqn

            scores, _, avg_max_q, grad_updates_at_score, loss_histories = algo_func(
                config=run_config,
                n_episodes=n_episodes,
                record_name=record_name,
                run_type=run_type,
                study_name=study_name,
                seed=seed
            )

            all_seed_scores.append(scores)
            all_seed_q_vals.append(avg_max_q)
            all_seed_grad_scores.append((grad_updates_at_score, scores))
            all_seed_ev.append(loss_histories.get('ev', []))
            all_seed_actor_loss.append(loss_histories.get('actor', []))
            all_seed_critic_loss.append(loss_histories.get('critic', []))
            all_seed_entropy_loss.append(loss_histories.get('entropy', []))
            all_seed_raw_entropy.append(loss_histories.get('raw_entropy', []))
            all_seed_actor_grad.append(loss_histories.get('actor_grad_norm', []))
            all_seed_critic_grad.append(loss_histories.get('critic_grad_norm', []))
            all_seed_ep_len.append(loss_histories.get('ep_len', []))

        results[exp_name] = all_seed_scores
        q_results[exp_name] = all_seed_q_vals
        grad_step_results[exp_name] = all_seed_grad_scores
        ev_results[exp_name] = all_seed_ev
        actor_loss_results[exp_name] = all_seed_actor_loss
        critic_loss_results[exp_name] = all_seed_critic_loss
        entropy_loss_results[exp_name] = all_seed_entropy_loss
        raw_entropy_results[exp_name] = all_seed_raw_entropy
        actor_grad_results[exp_name] = all_seed_actor_grad
        critic_grad_results[exp_name] = all_seed_critic_grad
        ep_len_results[exp_name] = all_seed_ep_len
        print(f"\n--- Finished: {exp_name} ---")

    print(f"\nAblation study finished in {(time.time() - timer)/60:.2f} minutes.")

    # --- Plotting the comparison ---
    print("\nGenerating comparison plots...")

    show_plots = base_config.save_parameters.get('show_plots', True)
    plot_ablation_statistics(
        results_dict=results,
        title='Ablation Study: Agent Performance vs Episode (Mean & Std Dev)',
        y_label='Average Score',
        output_path=f"{study_summary_dir}/scores_vs_episodes.png",
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

    # --- Second plot: score vs gradient updates (normalises for update-frequency differences) ---
    # Find the common gradient-update range across all runs and interpolate each seed onto it.
    max_grad_updates = max(
        seed_data[0][-1]
        for seed_list in grad_step_results.values()
        for seed_data in seed_list
        if seed_data[0]
    )
    common_grid = np.arange(0, max_grad_updates + 1)

    grad_results_interpolated = {}
    for exp_name, seed_list in grad_step_results.items():
        interp_runs = []
        for grad_updates_at_score, scores in seed_list:
            if len(grad_updates_at_score) > 1:
                interp_scores = np.interp(common_grid, grad_updates_at_score, scores)
            elif grad_updates_at_score:
                interp_scores = np.full_like(common_grid, scores[0], dtype=float)
            else:
                interp_scores = np.zeros_like(common_grid, dtype=float)
            interp_runs.append(interp_scores.tolist())
        grad_results_interpolated[exp_name] = interp_runs

    plot_ablation_statistics(
        results_dict=grad_results_interpolated,
        title='Ablation Study: Agent Performance vs Gradient Updates (Mean & Std Dev)',
        y_label='Average Score',
        output_path=f"{study_summary_dir}/scores_vs_grad_updates.png",
        win_condition=env_config.win_condition,
        show_plots=show_plots
    )

    # EV plot — algorithms with no critic return [] and are skipped automatically.
    plot_ablation_statistics(
        results_dict=ev_results,
        title='Ablation Study: Critic Explained Variance vs Episode',
        y_label='Explained Variance',
        output_path=f"{study_summary_dir}/explained_variance.png",
        show_plots=show_plots
    )

    # Combined loss plot per A2C experiment (all three weighted components on one axes).
    for exp_name in results:
        if not any(run for run in actor_loss_results.get(exp_name, [[]])):
            continue  # skip REINFORCE/DQN (no loss tracking)
        safe = (exp_name.replace(' ', '_').replace('=', '')
                        .replace('(', '').replace(')', ''))
        plot_ablation_statistics(
            results_dict={
                'Actor Loss':             actor_loss_results[exp_name],
                'Critic Loss (×weight)':  critic_loss_results[exp_name],
                'Entropy Loss (×weight)': entropy_loss_results[exp_name],
            },
            title=f'Loss Components: {exp_name}',
            y_label='Loss Magnitude',
            output_path=f"{study_summary_dir}/loss_components_{safe}.png",
            show_plots=show_plots
        )

    # Aggregate diagnostic plots across all experiments.
    plot_ablation_statistics(
        results_dict=raw_entropy_results,
        title='Ablation Study: Policy Entropy H(π) vs Episode',
        y_label='Entropy (nats)',
        output_path=f"{study_summary_dir}/policy_entropy.png",
        show_plots=show_plots
    )
    plot_ablation_statistics(
        results_dict=actor_grad_results,
        title='Ablation Study: Actor Gradient Norm vs Episode',
        y_label='L2 Norm (pre-clip)',
        output_path=f"{study_summary_dir}/actor_grad_norm.png",
        show_plots=show_plots
    )
    plot_ablation_statistics(
        results_dict=critic_grad_results,
        title='Ablation Study: Critic Gradient Norm vs Episode',
        y_label='L2 Norm (pre-clip)',
        output_path=f"{study_summary_dir}/critic_grad_norm.png",
        show_plots=show_plots
    )
    plot_ablation_statistics(
        results_dict=ep_len_results,
        title='Ablation Study: Episode Length vs Episode',
        y_label='Steps per Episode',
        output_path=f"{study_summary_dir}/episode_length.png",
        show_plots=show_plots
    )

if __name__ == '__main__':
    run_ablation_study()