import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from agent import Agent


current_version = "1-1-0"

# 0. decide which model to run
saved_model_name = f"{current_version}_demo_250"
# saved_model_name = f"{current_version}_demo_best" 
# 1. Initialize environment with RGB rendering (required for video)
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# 2. Wrap the environment to record video
# This will save the MP4 into a folder called "video"
env = RecordVideo(
    env, 
    video_folder="./video", 
    episode_trigger=lambda x: True,
    name_prefix=saved_model_name
)

# 3. Initialize the agent (architecture must match our training setup)
agent = Agent(state_size=8, action_size=4, seed=0)

# 4. Load the trained "brain" weights from our checkpoint file
# We use map_location='cpu' so it loads safely even if you trained on a GPU
agent.qnetwork_local.load_state_dict(torch.load(saved_model_name+'.pth', map_location=torch.device('cpu'), weights_only=True))
print("Trained brain loaded successfully! Action rolling...")

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

print(f"Landing complete! Final Score: {score:.2f}")

# 6. Close the environment to finalize and save the MP4 file
env.close()
print("Video saved in the './video' folder. Go check it out!")