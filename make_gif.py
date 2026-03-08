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

def make_gif():
    """
    Records videos for the 4 ablation agents and stitches them into a 2x2 GIF.
    Make sure you have moviepy installed: pip install moviepy
    """
    # --- Configuration ---
    config = OmegaConf.load("config.yaml")
    study_name = config.ablation_study.ablation_name
    first_seed = config.ablation_study.seeds[0]
    version_str = config.project.version.replace('.', '-')
    run_type = 'ablation'
    
    configs_to_render = [
        {'name': 'Full DQN (Buffer, Target)', 'suffix': 'Full_DQN_Buffer_Target'},
        {'name': 'No Replay Buffer', 'suffix': 'No_Replay_Buffer'},
        {'name': 'No Target Network', 'suffix': 'No_Target_Network'},
        {'name': 'Naive DQN (No Buffer, No Target)', 'suffix': 'Naive_DQN_No_Buffer_No_Target'},
    ]

    generated_clips = []
    
    # Path to the summary folder for this study, where the GIF will be saved
    study_summary_dir = f"raw_results/{version_str}/{run_type}/{study_name}"
    os.makedirs(study_summary_dir, exist_ok=True)

    # --- 1. Record a video for each agent ---
    for cfg in configs_to_render:
        print(f"--- Recording video for: {cfg['name']} ---")
        
        # Construct the path to the model file for the first seed
        record_name_base = f"{study_name}_{cfg['suffix']}"
        record_name = f"{record_name_base}_seed{first_seed}"
        model_dir = f"raw_results/{version_str}/{run_type}/{record_name}"
        model_path = os.path.join(model_dir, f"{record_name}_local_best.pth")
        # model_path = os.path.join(model_dir, f"{record_name}_last.pth")

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
        agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Run one episode
        state, _ = env.reset(seed=first_seed)
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
        # Landing bonus is +100, crashing is -100. We check if the final event was positive.
        cfg['landed_safely'] = terminated and not truncated and last_reward > 50
        
        env.close()
        
        # Create clip from frames
        clip = ImageSequenceClip(frames, fps=50)
        generated_clips.append(clip)

    # --- 2. Stitch videos into a 2x2 GIF ---
    print("\n--- Creating 2x2 GIF from videos ---")
    
    # Load raw clips first to determine max duration
    max_duration = max(c.duration for c in generated_clips)
    final_duration = max_duration + 0.5

    # Apply freeze and add text labels
    clips_with_text = []
    for i, clip in enumerate(generated_clips):
        landed = configs_to_render[i].get('landed_safely', False)
        border_color = (0, 255, 0) if landed else (255, 0, 0)

        # Freeze the last frame to extend to final_duration
        freeze_dur = final_duration - clip.duration
        if freeze_dur > 0:
            clip = clip.with_effects([vfx.Freeze(t='end', freeze_duration=freeze_dur)])
        
        clip = clip.with_effects([vfx.Margin(left=2, right=2, top=2, bottom=2)])
        # 1. Playing part (Black border)
        playing_part = clip.with_effects([vfx.Margin(left=2, right=2, top=2, bottom=2, color=(0,0,0))])
        
        # 2. Frozen part (Colored border)
        # We use max(0, ...) to avoid error on extremely short clips, though unlikely here
        last_frame = clip.get_frame(max(0, clip.duration - 0.05))
        frozen_part = ImageClip(last_frame).with_duration(freeze_dur)
        frozen_part = frozen_part.with_effects([vfx.Margin(left=2, right=2, top=2, bottom=2, color=border_color)])
        
        full_clip = concatenate_videoclips([playing_part, frozen_part])
        
        txt_clip = TextClip(text=configs_to_render[i]['name'], font_size=20, color='white', bg_color='black').with_position(('center', 'top')).with_duration(final_duration)
        clips_with_text.append(CompositeVideoClip([full_clip, txt_clip]))
        
    final_clip = clips_array([clips_with_text[:2], clips_with_text[2:]])
    gif_path = os.path.join(study_summary_dir, f"{study_name}_results.gif")
    final_clip.write_gif(gif_path, fps=20)
    
    print(f"\nSuccess! GIF saved to {gif_path}")

if __name__ == '__main__':
    make_gif()