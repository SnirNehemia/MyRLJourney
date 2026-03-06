import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo
from agent import Agent
from omegaconf import OmegaConf
import os

config = OmegaConf.load("config.yaml")

full_run_name = config.project.version.replace(".", "-") + "_"+ config.save_parameters.run_name

folder_path = f"raw_results/{full_run_name}"

prefix = full_run_name
suffix = '.pth'
files = [
    f for f in os.listdir(folder_path) 
    if f.startswith(prefix) and f.endswith(suffix)
]

for run_name in files:
    # 1. Initialize environment with RGB rendering (required for video)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    # 2. Wrap the environment to record video
    # This will save the MP4 into the run's result folder
    env = RecordVideo(
        env,
        video_folder=folder_path,
        episode_trigger=lambda x: True,
        name_prefix=run_name.replace(".pth", "")
    )

    # 3. Initialize the agent (architecture must match our training setup)
    agent = Agent(state_size=8, action_size=4, config=config, seed=0)

    # 4. Load the trained "brain" weights from our checkpoint file
    # We use map_location='cpu' so it loads safely even if you trained on a GPU
    agent.qnetwork_local.load_state_dict(torch.load(f'{folder_path}/{run_name}', map_location=torch.device('cpu'), weights_only=True))
    print(f"{run_name} brain loaded successfully! Action rolling...")

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

    # 6. Close the environment to finalize and save the MP4 file
    env.close()
    print("\tVideo saved in the './raw_video' folder. Go check it out!")

print('Done with: ',*files,sep='\n')