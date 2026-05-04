from omegaconf import OmegaConf
import gymnasium as gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import sys
import io
from moviepy import ImageSequenceClip, clips_array, TextClip, CompositeVideoClip
from PIL import Image, ImageDraw

# --- Path and Agent Setup ---
# Add parent directory to path to import the main DQN agent
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agent import Agent as DQNAgent # Main DQN agent from parent directory
from REINFORCE.agent import Agent as ReinforceAgent # REINFORCE agent from this folder

# --- Config and Device Setup ---
config = OmegaConf.load("config.yaml")
active_env_name = config.active_env
env_config = config.environments[active_env_name]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

output_dir = "comparison_results"
os.makedirs(output_dir, exist_ok=True)

def train_dqn(config, seed):
    """Trains the DQN agent."""
    print("\n--- Training DQN Agent ---")
    env = gym.make(active_env_name)
    agent = DQNAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, config=config, seed=seed)
    
    scores, scores_deque, best_avg_score = [], deque(maxlen=100), -np.inf
    model_path = os.path.join(output_dir, "dqn_best.pth")

    epsilon = config.training.exploration.epsilon.start

    for i_episode in range(1, config.training.n_episodes + 1):
        state, _ = env.reset(seed=seed + i_episode)
        score = 0
        for t in range(config.training.max_t):
            action = agent.act(state, exploration_param=epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, terminated or truncated, config.agent.tau)
            state = next_state
            score += reward
            if terminated or truncated: break
        
        scores_deque.append(score); scores.append(score)
        epsilon = max(config.training.exploration.epsilon.end, epsilon * config.training.exploration.epsilon.decay)
        avg_score = np.mean(scores_deque)
        
        print(f'\rDQN Episode {i_episode}\tAverage Score: {avg_score:.2f}', end="")
        if i_episode % 100 == 0: print(f'\rDQN Episode {i_episode}\tAverage Score: {avg_score:.2f}')
        
        if avg_score > best_avg_score and len(scores_deque) == 100:
            best_avg_score = avg_score
            torch.save(agent.qnetwork_local.state_dict(), model_path)

    env.close()
    return scores, model_path

def train_reinforce(config, seed):
    """Trains the REINFORCE agent."""
    print("\n--- Training REINFORCE Agent ---")
    env = gym.make(active_env_name)
    agent = ReinforceAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, config=config, seed=seed)

    scores, scores_deque, best_avg_score = [], deque(maxlen=100), -np.inf
    model_path = os.path.join(output_dir, "reinforce_best.pth")

    for i_episode in range(1, config.training.n_episodes + 1):
        state, _ = env.reset(seed=seed + i_episode)
        score = 0
        agent.rewards.clear(); agent.saved_log_probs.clear()

        for t in range(config.training.max_t):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            state = next_state
            score += reward
            if terminated or truncated: break
        
        scores_deque.append(score); scores.append(score)
        agent.finish_episode()
        avg_score = np.mean(scores_deque)

        print(f'\rREINFORCE Episode {i_episode}\tAverage Score: {avg_score:.2f}', end="")
        if i_episode % 100 == 0: print(f'\rREINFORCE Episode {i_episode}\tAverage Score: {avg_score:.2f}')

        if avg_score > best_avg_score and len(scores_deque) == 100:
            best_avg_score = avg_score
            torch.save(agent.policy_network.state_dict(), model_path)

    env.close()
    return scores, model_path

def plot_scores(dqn_scores, reinforce_scores):
    """Plots and saves the training progress of both agents."""
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(dqn_scores)), dqn_scores, label='DQN', alpha=0.7)
    plt.plot(np.arange(len(reinforce_scores)), reinforce_scores, label='REINFORCE', alpha=0.7)
    
    # Plot moving average
    dqn_moving_avg = np.convolve(dqn_scores, np.ones(100)/100, mode='valid')
    reinforce_moving_avg = np.convolve(reinforce_scores, np.ones(100)/100, mode='valid')
    plt.plot(np.arange(99, len(dqn_scores)), dqn_moving_avg, label='DQN (100-ep Avg)', linewidth=2)
    plt.plot(np.arange(99, len(reinforce_scores)), reinforce_moving_avg, label='REINFORCE (100-ep Avg)', linewidth=2)

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(f'DQN vs REINFORCE Training on {active_env_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "scores_comparison.png"))
    plt.close()

def create_visualization_plot(values, labels, chosen_idx, title, xlabel, plot_range, width, height):
    """Creates a bar chart for visualizing agent's network output (Q-values or probabilities)."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    colors = ['deepskyblue'] * len(labels)
    if chosen_idx < len(colors): colors[chosen_idx] = 'lime'

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, align='center', color=colors, height=0.5)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel, fontsize=9, color='lightgray')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
    if plot_range: ax.set_xlim(plot_range)
    fig.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=False, facecolor=fig.get_facecolor())
    buf.seek(0)
    plot_img = Image.open(buf).convert('RGB')
    plt.close(fig)
    return np.array(plot_img)

def run_episode_and_get_clip(agent_type, model_path, config, seed):
    """Runs one episode for a given agent and returns a moviepy clip."""
    print(f"  Generating video for {agent_type} agent...")
    env = gym.make(active_env_name, render_mode="rgb_array")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Load the correct agent and network
    if agent_type == 'DQN':
        agent = DQNAgent(state_size, action_size, config, seed=seed)
        agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=device))
        agent.qnetwork_local.eval()
    else: # REINFORCE
        agent = ReinforceAgent(state_size, action_size, config, seed=seed)
        agent.policy_network.load_state_dict(torch.load(model_path, map_location=device))
        agent.policy_network.eval()

    state, _ = env.reset(seed=seed)
    done = False
    score = 0
    frames = []
    action_labels = env_config.get('action_labels', [f'Action {i}' for i in range(action_size)])

    while not done:
        # --- Get Action and Network Output ---
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            if agent_type == 'DQN':
                action_values = agent.qnetwork_local(state_tensor).squeeze()
                action = np.argmax(action_values.cpu().numpy())
                viz_values = action_values.cpu().numpy()
                viz_title, viz_xlabel, viz_range = 'Action Q-Values', 'Q-Value', env_config.get('q_plot_range', None)
            else: # REINFORCE
                logits = agent.policy_network(state_tensor).squeeze()
                probs = torch.softmax(logits, dim=0)
                action = torch.distributions.Categorical(probs).sample().item()
                viz_values = probs.cpu().numpy()
                viz_title, viz_xlabel, viz_range = 'Action Probabilities', 'Probability', [0, 1]

        # --- Render Frames ---
        env_frame = env.render()
        pil_img = Image.fromarray(env_frame)
        draw = ImageDraw.Draw(pil_img)
        draw.text((10, 10), f"Score: {score:.1f}", fill=(255, 255, 255))
        
        # Create visualization plot
        plot_img = create_visualization_plot(viz_values, action_labels, action, viz_title, viz_xlabel, viz_range, 300, env_frame.shape[0])
        
        # Combine environment and plot
        combined_frame = np.hstack((np.array(pil_img), plot_img))
        frames.append(combined_frame)

        # --- Step Environment ---
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        score += reward

    env.close()
    return ImageSequenceClip(frames, fps=30)

def generate_comparison_gif(dqn_model_path, reinforce_model_path, config, seed):
    """Generates and saves a side-by-side GIF of the two trained agents."""
    if not os.path.exists(dqn_model_path) or not os.path.exists(reinforce_model_path):
        print("One or both model files not found. Skipping GIF generation.")
        return

    print("\n--- Generating Comparison GIF ---")
    dqn_clip = run_episode_and_get_clip('DQN', dqn_model_path, config, seed)
    reinforce_clip = run_episode_and_get_clip('REINFORCE', reinforce_model_path, config, seed)

    # Add text labels
    dqn_text = TextClip(text='DQN', font_size=30, color='white', bg_color='black').with_position(('center', 'top')).with_duration(dqn_clip.duration)
    reinforce_text = TextClip(text='REINFORCE', font_size=30, color='white', bg_color='black').with_position(('center', 'top')).with_duration(reinforce_clip.duration)

    dqn_clip_with_text = CompositeVideoClip([dqn_clip, dqn_text])
    reinforce_clip_with_text = CompositeVideoClip([reinforce_clip, reinforce_text])

    # Combine clips
    final_clip = clips_array([[dqn_clip_with_text, reinforce_clip_with_text]])
    gif_path = os.path.join(output_dir, "agent_comparison.gif")
    final_clip.write_gif(gif_path, fps=20)
    print(f"\nSuccess! GIF saved to {gif_path}")

if __name__ == '__main__':
    # Use a consistent seed for training runs
    training_seed = config.project.seed

    dqn_scores, dqn_model_path = train_dqn(config, training_seed)
    reinforce_scores, reinforce_model_path = train_reinforce(config, training_seed)
    
    plot_scores(dqn_scores, reinforce_scores)
    
    # Use a different, consistent seed for GIF generation to show a new trajectory
    gif_seed = training_seed + config.training.n_episodes + 1 
    generate_comparison_gif(dqn_model_path, reinforce_model_path, config, gif_seed)
