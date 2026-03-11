import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo
from agent import Agent
from omegaconf import OmegaConf
import os

config = OmegaConf.load("config.yaml")

def record_videos_for_main_run():
    """
    Records a video for the main training run specified in config.yaml.
    It looks for the model trained with the default seed.
    """
    active_env_name = config.active_env
    version_str = config.project.version.replace('.', '-')
    run_name = config.save_parameters.run_name
    seed = config.project.seed
    run_type = 'train'

    record_name = f"{run_name}_seed{seed}"
    model_folder = f"raw_results/{active_env_name}/{version_str}/{run_type}/{record_name}"
    
    if not os.path.exists(model_folder):
        print(f"ERROR: Model folder not found at '{model_folder}'")
        print("Please run train.py first to generate the models.")
        return

    # Find all model files in the folder
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.pth')]

    if not model_files:
        print(f"ERROR: No model files (.pth) found in '{model_folder}'")
        return

    print(f"Found {len(model_files)} models in '{model_folder}'. Generating videos...")

    for model_file in model_files:
        model_path = os.path.join(model_folder, model_file)
        
        # 1. Initialize environment with RGB rendering (required for video)
        env_config_video = config.environments[active_env_name]
        env_params_video = {}
        if 'lunar_params' in env_config_video:
            env_params_video = OmegaConf.to_container(env_config_video.lunar_params)
        env = gym.make(active_env_name, render_mode="rgb_array", **env_params_video)

        # 2. Wrap the environment to record video
        env = RecordVideo(
            env,
            video_folder=model_folder, # Save video in the same folder as the model
            episode_trigger=lambda x: True,
            name_prefix=model_file.replace('.pth', '')
        )

        # 3. Initialize the agent (architecture must match our training setup)
        env_config = config.environments[active_env_name]
        agent = Agent(state_size=env_config.state_size, action_size=env_config.action_size, config=config, seed=0)

        # 4. Load the trained "brain" weights from our checkpoint file
        agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"\n'{model_file}' brain loaded successfully! Rolling video...")

        # 5. Play one single game
        state, info = env.reset()
        score = 0
        done = False

        while not done:
            action = agent.act(state, eps=0.0)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            score += reward

        print(f"\tEpisode complete! Final Score: {score:.2f}")
        env.close()
        print(f"\tVideo saved in '{model_folder}'.")

if __name__ == '__main__':
    record_videos_for_main_run()