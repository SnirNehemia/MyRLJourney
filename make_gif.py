import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo
from moviepy import VideoFileClip, clips_array, TextClip, CompositeVideoClip, vfx
import os
import glob
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
    run_name_prefix = "ablation_study"
    
    configs_to_render = [
        {'name': 'Full DQN (Buffer, Target)', 'suffix': 'Full_DQN_Buffer_Target'},
        {'name': 'No Replay Buffer', 'suffix': 'No_Replay_Buffer'},
        {'name': 'No Target Network', 'suffix': 'No_Target_Network'},
        {'name': 'Naive DQN (No Buffer, No Target)', 'suffix': 'Naive_DQN_No_Buffer_No_Target'},
    ]

    video_files = []
    base_video_folder = "raw_videos"
    base_gif_folder = "raw_gifs"
    temp_video_folder = os.path.join(base_video_folder, "temp_videos_for_gif")
    if os.path.exists(temp_video_folder):
        shutil.rmtree(temp_video_folder) # Clean up previous runs
    os.makedirs(temp_video_folder, exist_ok=True)
    os.makedirs(base_gif_folder, exist_ok=True)

    # --- 1. Record a video for each agent ---
    for cfg in configs_to_render:
        print(f"--- Recording video for: {cfg['name']} ---")
        
        # Construct the path to the model file
        record_name = f"{run_name_prefix}_{cfg['suffix']}"
        model_dir = f"raw_results/{config.project.version.replace('.', '-')}_{record_name}"
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
        
        env = RecordVideo(
            env,
            video_folder=temp_video_folder,
            episode_trigger=lambda x: x == 0, # Record the first episode
            name_prefix=cfg['suffix']
        )
        
        agent = Agent(state_size=8, action_size=4, config=run_config, seed=0)
        agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        # Run one episode
        state, _ = env.reset(seed=config.ablation_study.seed + 1)
        done = False
        while not done:
            action = agent.act(state, eps=0.0) # Use greedy policy
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        
        # The video is saved automatically when env is closed or reset
        env.close()
        
        # Find the created mp4 file and add it to our list
        mp4_file = glob.glob(os.path.join(temp_video_folder, f"{cfg['suffix']}*.mp4"))[0]
        video_files.append(mp4_file)
        print(f"Video saved: {mp4_file}")

    if len(video_files) != 4:
        print("Error: Did not generate 4 videos. Aborting GIF creation.")
        return

    # --- 2. Stitch videos into a 2x2 GIF ---
    print("\n--- Creating 2x2 GIF from videos ---")
    
    # Load raw clips first to determine max duration
    raw_clips = [VideoFileClip(f) for f in video_files]
    max_duration = max(c.duration for c in raw_clips)
    final_duration = max_duration + 0.5

    # Apply freeze and add text labels
    clips_with_text = []
    for i, clip in enumerate(raw_clips):
        # Freeze the last frame to extend to final_duration
        freeze_dur = final_duration - clip.duration
        if freeze_dur > 0:
            clip = clip.with_effects([vfx.Freeze(t='end', freeze_duration=freeze_dur)])
        
        clip = clip.with_effects([vfx.Margin(left=2, right=2, top=2, bottom=2)])
        txt_clip = TextClip(text=configs_to_render[i]['name'], font_size=20, color='white', bg_color='black').with_position(('center', 'top')).with_duration(final_duration)
        clips_with_text.append(CompositeVideoClip([clip, txt_clip]))
        
    final_clip = clips_array([clips_with_text[:2], clips_with_text[2:]])
    gif_path = os.path.join(base_gif_folder, f"{config.save_parameters.run_name}_ablation_study_results.gif")
    final_clip.write_gif(gif_path, fps=20)
    
    print(f"\nSuccess! GIF saved to {gif_path}")
    # shutil.rmtree(temp_video_folder) # Clean up temp files

if __name__ == '__main__':
    make_gif()