import gymnasium as gym

# Initialize the Lunar Lander environment
# render_mode="human" allows you to see the window pop up
env = gym.make("LunarLander-v3", render_mode="human")

observation, info = env.reset()

for _ in range(100):
    # Take a random action (0, 1, 2, or 3) just to see what happens
    action = env.action_space.sample()  
    
    # Step the environment forward
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()