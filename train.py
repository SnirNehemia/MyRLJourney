import gymnasium as gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
from agent import Agent
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")
__version__ = config.project.version

record_name = __version__.replace(".", "-") + "_"+ config.video.record_name
# 1. Initialize the Environment and Agent
env = gym.make('LunarLander-v3',
               enable_wind=False, wind_power=15.0, turbulence_power=1.5)
# State size is 8 (coordinates, velocity, angle, etc.)
# Action size is 4 (engines: none, left, main, right)
agent = Agent(state_size=8, action_size=4, seed=config.project.seed)

def dqn(n_episodes=config.training.n_episodes, max_t=config.training.max_t,
         eps_start=config.training.eps_start, eps_end=config.training.eps_end, eps_decay=config.training.eps_decay,
         win_condition=config.training.win_condition):
    """Deep Q-Learning loop."""
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores for our win condition
    eps = eps_start                    # initialize epsilon
    time_ref = time.time()
    best_score = -float('inf')

    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset()
        score = 0
        
        for t in range(max_t):
            # 1. The agent picks an action
            action = agent.act(state, eps)
            
            # 2. The environment reacts
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            reward = modify_reward(reward, action)

            # 3. The agent learns from the result
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            if done:
                break 
                
        # Save scores
        scores_window.append(score)       
        scores.append(score)              
        
        # Decrease epsilon (less random exploration over time)
        eps = max(eps_end, eps_decay * eps) 
        
        # Print progress to the terminal
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.0f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tTime: {time.time() - time_ref:.2f} seconds = {(time.time() - time_ref)/60:.2f} minutes')
            
        # Save whenever we hit a new high score
        current_avg = np.mean(scores_window)
        if current_avg > best_score and i_episode > 1000:
            best_score = current_avg
            torch.save(agent.qnetwork_local.state_dict(), f'results/{record_name}_local_best.pth')

        if i_episode % 250 == 0:
            torch.save(agent.qnetwork_local.state_dict(), f'results/{record_name}_{i_episode}-eps.pth')

        # Check Win Condition
        if current_avg >= win_condition:
            print(f'\nEnvironment solved in {i_episode-100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            # Save the trained neural network weights!
            torch.save(agent.qnetwork_local.state_dict(), f'results/{record_name}_best.pth')
            break
            
    return scores

def modify_reward(reward, action):
    # Custom reward shaping to encourage unique landing behavior
    # Discourage the agent from using the main engine (action 2), out of interest of seeing more creative solutions. This is optional and can be adjusted based on your preferences.
    # if action == 2:
    #     reward -= 0.05

    return reward

if __name__ == '__main__':
    print("Starting Training! This might take 5 to 15 minutes depending on your CPU...")
    scores = dqn()

    # Plot the learning curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Agent Training Curve')
    plt.grid()
    plt.show()