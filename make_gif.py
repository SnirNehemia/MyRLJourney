import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo
from moviepy import VideoFileClip, clips_array, TextClip, CompositeVideoClip, vfx, ImageClip, concatenate_videoclips, ImageSequenceClip
import numpy as np
from PIL import Image, ImageDraw
import os
import shutil

from agent import Agent
from omegaconf import OmegaConf

def make_gifs_for_study():
    """
    Records videos for the ablation agents and stitches them into 2x2 GIFs.
    This is done for the first `n_gifs` seeds defined in the ablation study.
    """
    # --- Configuration ---
    config = OmegaConf.load("config.yaml")
    study_name = config.ablation_study.ablation_name
    ablation_seeds = config.ablation_study.seeds
    n_gifs = config.save_parameters.get('n_gifs', 1)
    version_str = config.project.version.replace('.', '-')
    run_type = 'ablation'
    
    # Path to the summary folder for this study, where the GIF will be saved
    study_summary_dir = f"raw_results/{version_str}/{run_type}/{study_name}"
    os.makedirs(study_summary_dir, exist_ok=True)

    # --- Determine which configurations to render ---
    configs_to_render_template = []
    is_sweep = config.ablation_study.get('sweep', {}).get('enabled', False)
    if is_sweep:
        sweep_config = config.ablation_study.sweep
        param_to_sweep = sweep_config.parameter
        sweep_values = sweep_config['list_values']
        param_name_for_label = param_to_sweep.split('.')[-1]
        for value in sweep_values:
            name = f"{param_name_for_label}={value}"
            suffix = name.replace('=', '_')
            configs_to_render_template.append({'name': name, 'suffix': suffix})
    else:
        configs_to_render_template = [
            {'name': 'Full DQN (Buffer, Target)', 'suffix': 'Full_DQN_Buffer_Target'},
            {'name': 'No Replay Buffer', 'suffix': 'No_Replay_Buffer'},
            {'name': 'No Target Network', 'suffix': 'No_Target_Network'},
            {'name': 'Naive DQN (No Buffer, No Target)', 'suffix': 'Naive_DQN_No_Buffer_No_Target'},
        ]

    for i in range(n_gifs):
        seed_to_use = i
        print(f"\n--- Generating GIF for seed: {seed_to_use} ---")

        generated_clips = []
        configs_this_run = [cfg.copy() for cfg in configs_to_render_template]

        # --- 1. Record a video for each agent ---
        for cfg in configs_this_run:
            print(f"--- Recording video for: {cfg['name']} ---")
            
            # Construct the path to the model file for the current seed
            record_name_base = f"{study_name}_{cfg['suffix']}"
            record_name = f"{record_name_base}_seed{seed_to_use}"
            model_dir = f"raw_results/{version_str}/{run_type}/{record_name}"
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
            env = gym.make("LunarLander-v3", render_mode="rgb_array")
            
            agent = Agent(state_size=8, action_size=4, config=run_config, seed=0)
            agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
            
            # Run one episode
            state, _ = env.reset(seed=seed_to_use)
            done = False
            score = 0
            frames = []
            last_reward = 0

            t = 0
            while not done and t < config.training.max_t:
                t += 1
                # 1. Render frame
                frame = env.render()
                
                # 2. Draw current score on frame
                pil_img = Image.fromarray(frame)
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 10), f"Score: {score:.1f}", fill=(255, 255, 255))
                frames.append(np.array(pil_img))

                # 3. Step environment
                action = agent.act(state, eps=0.0) # Use greedy policy
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                score += reward
                last_reward = reward
            
            # Determine if landed successfully (Terminated naturally + Positive reward for landing)
            cfg['landed_safely'] = terminated and not truncated and last_reward > 50
            
            env.close()
            
            # Create clip from frames
            clip = ImageSequenceClip(frames, fps=50)
            generated_clips.append(clip)

        if not generated_clips:
            print(f"No clips generated for seed {seed_to_use}. Skipping GIF creation.")
            continue

        # --- 2. Stitch videos into a 2x2 GIF ---
        print(f"\n--- Creating GIF for seed {seed_to_use} ---")
        
        max_duration = max(c.duration for c in generated_clips) if generated_clips else 0
        final_duration = max_duration + 0.5

        # Apply freeze and add text labels
        clips_with_text = []
        for i, clip in enumerate(generated_clips):
            landed = configs_this_run[i].get('landed_safely', False)
            border_color = (0, 255, 0) if landed else (255, 0, 0)

            freeze_dur = final_duration - clip.duration
            
            # 1. Playing part (Black border)
            playing_part = clip.with_effects([vfx.Margin(left=2, right=2, top=2, bottom=2, color=(0,0,0))])
            
            # 2. Frozen part (Colored border)
            if freeze_dur > 0:
                last_frame = clip.get_frame(clip.duration - 0.01)
                frozen_part = ImageClip(last_frame, duration=freeze_dur)
                frozen_part = frozen_part.with_effects([vfx.Margin(left=2, right=2, top=2, bottom=2, color=border_color)])
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
        gif_path = os.path.join(study_summary_dir, f"{study_name}_seed{seed_to_use}_results.gif")
        final_clip.write_gif(gif_path, fps=20)
        
        print(f"\nSuccess! GIF saved to {gif_path}")

if __name__ == '__main__':
    make_gifs_for_study()