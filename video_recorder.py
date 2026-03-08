import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo
from agent import Agent
from omegaconf import OmegaConf
import os

config = OmegaConf.load("config.yaml")

def record_main_training_run():
    """
    Records a video for the main training run specified in config.yaml.
    It looks for the model trained with the default seed.
    """
    version_str = config.project.version.replace('.', '-')
    run_name = config.save_parameters.run_name
    seed = config.project.seed
    run_type = 'train'

    record_name = f"{run_name}_seed{seed}"
    model_folder = f"raw_results/{version_str}/{run_type}/{record_name}"
    
    # We'll use the final saved model for recording
    model_file = f"{record_name}_last.pth"
    model_path = os.path.join(model_folder, model_file)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at '{model_path}'")
        print("Please run train.py first to generate the model.")
        return
    
    # 1. Initialize environment with RGB rendering (required for video)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # 2. Wrap the environment to record video
    env = RecordVideo(
        env,
        video_folder=model_folder, # Save video in the same folder as the model
        episode_trigger=lambda x: True,
        name_prefix=record_name
    )

    # 3. Initialize the agent (architecture must match our training setup)
    agent = Agent(state_size=8, action_size=4, config=config, seed=0)

    # 4. Load the trained "brain" weights from our checkpoint file
    agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    print(f"'{model_file}' brain loaded successfully! Rolling video...")

    # 5. Play one single game
    state, info = env.reset()
    score = 0
    done = False

    while not done:
        # Notice eps=0.0! We want 0% exploration and 100% exploitation.
        # The agent relies entirely on what it learned.
        action = agent.act(state, eps=0.0)
        
        # Take the action in the environment
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward

    print(f"\tLanding complete! Final Score: {score:.2f}")

    env.close()
    print(f"\tVideo saved in '{model_folder}'.")

if __name__ == '__main__':
    record_main_training_run()