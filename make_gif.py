import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo
from moviepy import (VideoFileClip, clips_array, TextClip, CompositeVideoClip, 
                            ImageClip, concatenate_videoclips, ImageSequenceClip) # Removed vfx
import numpy as np
from PIL import Image, ImageDraw
import os
import shutil
import io
import matplotlib.pyplot as plt

from agent import Agent, device
from omegaconf import OmegaConf

def add_border_to_numpy_frame(frame_array, border_size, color):
    """Adds a border of specified size and color to a numpy image array.
    Color should be an RGB tuple (e.g., (0, 0, 0) for black).
    """
    h, w, c = frame_array.shape
    # Create a new array for the bordered frame, initialized with the border color
    bordered_frame = np.full((h + 2 * border_size, w + 2 * border_size, c), color, dtype=frame_array.dtype)
    # Paste the original frame into the center
    bordered_frame[border_size:border_size + h, border_size:border_size + w] = frame_array
    return bordered_frame

def create_saliency_plot(saliency_values, labels, width, height):
    """Creates a bar chart image from saliency values using a dark theme."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    
    # Highlight the bar with the highest value
    colors = ['cyan'] * len(labels)
    if saliency_values.any(): # Ensure there's a max value to highlight
        max_idx = np.argmax(saliency_values)
        colors[max_idx] = 'magenta'
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, saliency_values, align='center', color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '0.5', '1.0'], fontsize=8)
    ax.set_xlim(0, 1)
    
    ax.set_xlabel('Normalized Importance', fontsize=9, color='lightgray')
    ax.set_title('Input Saliency', fontsize=11, fontweight='bold', pad=10)
    
    fig.tight_layout(pad=1.5)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=False, facecolor=fig.get_facecolor())
    buf.seek(0)
    plot_img = Image.open(buf).convert('RGB')
    plot_array = np.array(plot_img)
    plt.close(fig)
    
    return plot_array

def create_q_value_plot(q_values, action_labels, chosen_action_idx, width, height, use_fake_actions=False, real_action_size=4):
    """Creates a bar chart for Q-values of each action."""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)

    # --- Process Q-values and labels for fake actions ---
    if use_fake_actions and len(q_values) > real_action_size:
        real_q_values = q_values[:real_action_size]
        fake_q_values = q_values[real_action_size:]
        avg_fake_q = np.mean(fake_q_values)
        
        processed_q_values = np.append(real_q_values, avg_fake_q)
        
        processed_labels = action_labels[:real_action_size]
        processed_labels.append('Fake Actions')
        
        # Adjust chosen action index if it was a fake action
        if chosen_action_idx >= real_action_size:
            processed_chosen_action_idx = real_action_size # The last bar is now 'Fake Actions'
        else:
            processed_chosen_action_idx = chosen_action_idx
    else:
        processed_q_values = q_values
        processed_labels = action_labels
        processed_chosen_action_idx = chosen_action_idx

    colors = ['deepskyblue'] * len(processed_labels)
    colors[processed_chosen_action_idx] = 'lime'

    y_pos = np.arange(len(processed_labels))
    bars = ax.barh(y_pos, processed_q_values, align='center', color=colors, height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(processed_labels, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel('Q-Value', fontsize=9, color='lightgray')
    ax.set_title('Action Q-Values', fontsize=11, fontweight='bold', pad=10)
    
    fixed_min_q = -200
    fixed_max_q = 200
    ax.set_xlim(fixed_min_q, fixed_max_q)

    fig.tight_layout(pad=1.5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=False, facecolor=fig.get_facecolor())
    buf.seek(0)
    plot_img = Image.open(buf).convert('RGB')
    plot_array = np.array(plot_img)
    plt.close(fig)

    return plot_array

def make_gifs_for_study(model_seed=0, run_type='ablation'):
    # --- Configuration ---
    config = OmegaConf.load("config.yaml")
    active_env_name = config.active_env
    study_name = config.ablation_study.ablation_name
    n_gifs = config.save_parameters.get('n_gifs', 1)
    version_str = config.project.version.replace('.', '-')
    add_saliency = config.save_parameters.get('add_saliency_to_gif', False)
    run_type = run_type if run_type is not None else 'ablation'
    
    # Path to the summary folder for this study, where the GIF will be saved
    study_summary_dir = f"raw_results/{active_env_name}/{version_str}/{run_type}/{study_name}"
    os.makedirs(study_summary_dir, exist_ok=True)

    # --- Determine which configurations to render ---
    configs_to_render_template = []
    study_type = config.ablation_study.get('study_type', 'component')

    if study_type == 'sweep':
        sweep_config = config.ablation_study.sweep
        param_to_sweep = sweep_config.parameter
        sweep_values = sweep_config['list_values']
        param_name_for_label = param_to_sweep.split('.')[-1]
        for value in sweep_values:
            name = f"{param_name_for_label}={value}"
            suffix = name.replace('=', '_').replace('.', 'p')
            configs_to_render_template.append({'name': name, 'suffix': suffix})
    elif study_type == 'dqn_variants':
        configs_to_render_template = [
            {'name': 'DQN (No Target)', 'suffix': 'DQN_No_Target'},
            {'name': 'DQN (With Target)', 'suffix': 'DQN_With_Target'},
            {'name': 'Double DQN', 'suffix': 'Double_DQN'},
        ]
    else: # 'component'
        configs_to_render_template = [
            {'name': 'Full DQN (Buffer, Target)', 'suffix': 'Full_DQN_Buffer_Target'},
            {'name': 'No Replay Buffer', 'suffix': 'No_Replay_Buffer'},
            {'name': 'No Target Network', 'suffix': 'No_Target_Network'},
            {'name': 'Naive DQN (No Buffer, No Target)', 'suffix': 'Naive_DQN_No_Buffer_No_Target'},
        ]

    for i in range(min(n_gifs, len(config.ablation_study.seeds))):
        model_seed = config.ablation_study.seeds[i]
        env_seed = model_seed # Use the same seed for model and env for consistency
        print(f"\n--- Generating GIF for seed: {env_seed} ---")

        generated_clips = []
        configs_this_run = [cfg.copy() for cfg in configs_to_render_template]

        # --- 1. Record a video for each agent ---
        for cfg in configs_this_run:
            print(f"--- Recording video for: {cfg['name']} ---")
            
            # Construct the path to the model file for the current seed
            record_name_base = f"{study_name}_{cfg['suffix']}"
            record_name = f"{record_name_base}_seed{model_seed}"
            model_dir = f"raw_results/{active_env_name}/{version_str}/{run_type}/{record_name}"
            model_path = os.path.join(model_dir, f"{record_name}_local_best.pth")

            # Load the specific config file that was used for this training run
            run_config_path = os.path.join(model_dir, "run_config.yaml")
            if os.path.exists(run_config_path):
                run_config = OmegaConf.load(run_config_path)
            else:
                run_config = config # Fallback to base config if not found

            if not os.path.exists(model_path):
                print(f"Warning: Model file not found at {model_path}. Skipping.")
                continue

            # Initialize environment and agent
            env_config_gif = run_config.environments[run_config.active_env]
            env_params_gif = {}
            if 'lunar_params' in env_config_gif:
                env_params_gif = OmegaConf.to_container(env_config_gif.lunar_params)
            env = gym.make(run_config.active_env, render_mode="rgb_array", **env_params_gif)
            
            env_config = run_config.environments[run_config.active_env]

            # --- Fake Actions Logic ---
            real_action_size = env_config.action_size
            agent_action_size = real_action_size
            use_fake_actions = (run_config.active_env == "LunarLander-v3" and 
                                env_config.get('use_fake_actions', False))
            if use_fake_actions:
                num_fake = env_config.get('num_fake_actions', 6)
                agent_action_size += num_fake

            agent = Agent(state_size=env_config.state_size, action_size=agent_action_size, config=run_config, seed=0)
            agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            
            # Get state labels for saliency plot if applicable
            saliency_labels = None
            action_labels = None
            if add_saliency:
                saliency_labels = env_config.get('state_labels', [f'Input {i}' for i in range(env_config.state_size)])
                action_labels = env_config.get('action_labels', [f'Action {i}' for i in range(real_action_size)])
                if use_fake_actions:
                    num_fake = env_config.get('num_fake_actions', 6)
                    for i in range(num_fake):
                        action_labels.append(f'Fake {i+1}')

            # Run one episode
            state, _ = env.reset(seed=env_seed)
            done = False
            score = 0
            frames = []
            last_reward = 0

            t = 0
            while not done and t < config.training.max_t:
                t += 1
                
                # 1. Render environment frame and draw score
                env_frame = env.render()
                pil_img = Image.fromarray(env_frame)
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 10), f"Score: {score:.1f}", fill=(255, 255, 255))
                frame_with_score = np.array(pil_img)

                # Add black border to the environment frame
                bordered_env_frame = add_border_to_numpy_frame(frame_with_score, 2, (0, 0, 0))
                frame_h = bordered_env_frame.shape[0]

                # --- Get Action and Saliency ---
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
                state_tensor.requires_grad = True

                agent.qnetwork_local.eval()
                action_values = agent.qnetwork_local(state_tensor)
                agent.qnetwork_local.train()

                agent_action = torch.argmax(action_values.detach()).item()

                # Map agent action to real environment action
                env_action = agent_action
                if use_fake_actions and agent_action >= real_action_size:
                    env_action = 0 # Map all fake actions to 'No-Op'


                saliency_plot_img, q_plot_img = None, None

                if add_saliency and saliency_labels:
                    max_q_value = action_values[0, env_action]
                    
                    agent.qnetwork_local.zero_grad()
                    max_q_value.backward()
                    
                    saliency = state_tensor.grad.abs().squeeze(0).cpu().numpy()
                    
                    # normalize the saliency values
                    if np.max(saliency) > 0:
                        saliency /= np.max(saliency)
                    else:
                        saliency = np.zeros_like(saliency)
                    
                    # Calculate dynamic plot height based on environment frame
                    plot_h = frame_h // 2
                    plot_h_rem = frame_h % 2
                    
                    # Saliency plot
                    saliency_plot_img = create_saliency_plot(saliency, saliency_labels, 300, plot_h)

                    # Create Q-value plot
                    q_values_np = action_values.detach().cpu().squeeze(0).numpy()
                    q_plot_img = create_q_value_plot(q_values_np, action_labels, agent_action, 300, plot_h + plot_h_rem,
                                                     use_fake_actions=use_fake_actions, real_action_size=real_action_size)

                # 2. Combine with saliency plot if it exists
                if saliency_plot_img is not None and q_plot_img is not None:
                    # Stack plots vertically, then attach to the side of the env render
                    plots_panel = np.vstack((saliency_plot_img, q_plot_img))
                    combined_frame = np.hstack((bordered_env_frame, plots_panel))
                    frames.append(combined_frame)
                else:
                    frames.append(bordered_env_frame) # Still add border even without saliency

                # 3. Step environment with the chosen action
                state, reward, terminated, truncated, _ = env.step(env_action)
                done = terminated or truncated
                score += reward
                last_reward = reward
            
            # Determine if landed successfully (Terminated naturally + Positive reward for landing)
            env_config = run_config.environments[run_config.active_env]
            if run_config.active_env == "LunarLander-v3":
                # For LunarLander, a safe landing is a non-crash termination with a positive final reward
                cfg['succeeded'] = terminated and not truncated and last_reward > 50
            else:
                # For other envs like Acrobot, success is defined by reaching the score threshold
                cfg['succeeded'] = score >= env_config.win_condition
            
            env.close()
            
            # Create clip from frames
            clip = ImageSequenceClip(frames, fps=50)
            generated_clips.append(clip)

        if not generated_clips:
            print(f"No clips generated for seed {model_seed}. Skipping GIF creation.")
            continue

        # --- 2. Stitch videos into a 2x2 GIF ---
        print(f"\n--- Creating GIF for seed {model_seed} ---")
        
        max_duration = max(c.duration for c in generated_clips) if generated_clips else 0
        final_duration = max_duration + 0.5

        # Apply freeze and add text labels
        clips_with_text = []
        for i, clip in enumerate(generated_clips):
            landed = configs_this_run[i].get('succeeded', False)
            border_color = (0, 255, 0) if landed else (255, 0, 0)

            freeze_dur = final_duration - clip.duration
            # The 'clip' already has the black border from the frame processing loop
            playing_part = clip 
            
            # 2. Frozen part (Colored border)
            if freeze_dur > 0:
                last_frame = clip.get_frame(clip.duration - 0.01)
                # Manually add the colored border to the last frame
                bordered_last_frame = add_border_to_numpy_frame(last_frame, 2, border_color)
                frozen_part = ImageClip(bordered_last_frame, duration=freeze_dur)
                
                full_clip = concatenate_videoclips([playing_part, frozen_part])
            else:
                full_clip = playing_part
            
            txt_clip = TextClip(text=configs_this_run[i]['name'], font_size=20, color='white', bg_color='black').with_position(('center', 'top')).with_duration(final_duration)
            clips_with_text.append(CompositeVideoClip([full_clip, txt_clip]))
            
        # Pad with black clips if necessary to make a 2x2 grid
        if len(clips_with_text) < 4:
            print(f"Warning: Only {len(clips_with_text)} clips were processed. Padding with black squares.")
            black_clip = ImageClip(np.zeros((1,1,3)), duration=final_duration)
            while len(clips_with_text) < 4:
                clips_with_text.append(black_clip)

        final_clip = clips_array([clips_with_text[:2], clips_with_text[2:]])
        gif_path = os.path.join(study_summary_dir, f"{study_name}_seed{model_seed}_comparison.gif")
        final_clip.write_gif(gif_path, fps=20)
        
        print(f"\nSuccess! GIF saved to {gif_path}")

if __name__ == '__main__':
    make_gifs_for_study()