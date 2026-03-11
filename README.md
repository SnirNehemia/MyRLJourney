# Deep Q-Network (DQN) Lunar Lander

![Version](https://img.shields.io/badge/version-1.3.1-success)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![Gymnasium](https://img.shields.io/badge/Gymnasium-RL%20Environment-lightgrey)

An implementation of a Deep Q-Network (DQN) from scratch in PyTorch to solve the Gymnasium environments. 

### Replay Buffer Length Effect (with wind and turbulence)
![alt text](gif/ReplayBuffer_sweep_EnvSeed4_ModelSeed(1).gif)


## Project Overview
The goal of this project is to learn how to train an RL agent.

This project implements the DQN algorithm using Pytorch, and on future version I will explore more advanced methods for this task.

It includes a thorough ablation study to study the effect of the concepts implemented here.

## Technical Implementation
* **Framework:** PyTorch
* **Action Selection:** $\epsilon$-greedy policy with decay.
* **Optimization:** Mean Squared Error (MSE) Loss with Adam Optimizer.
* **core concepts implemented:**
  * DQN 
  * DDQN

### Key RL Features Included:
1. **Experience Replay Buffer:** Breaks the correlation of sequential observations by randomly sampling past experiences (State, Action, Reward, Next State) to train the network.
2. **Fixed Q-Targets:** Utilizes a "Local" network for active playing/learning and a frozen "Target" network for calculating the expected future rewards. This prevents the moving target problem and stabilizes training.
3. **Double DQN:** Use the local network for action prediction and the target network as Q value estimator.

## Training Results & Analysis
 
Examples of lunard lander agent:

an example of score (total reward) vs. episode:
![alt text](images/buffer_size_ablation_1-2-7/scores_comparison.png)

and the testing performance:
![alt text](images/buffer_size_ablation_1-2-7/test_scores_violin.png)

## Discussions

Discussions for each major update should be in the discussion folder.

## Version history:
* 1.0.0 - standard DQN

   > Replay Buffer\
  Target Network

* 1.1.0 - QoL

  > Easier network modifications

* 1.2.0 - DDQN & config

  > Double DQN

  > Configuration file

* 1.2.1 - Experiments

  > Experiments script

* 1.2.7 - Ablation Study

  > Ablation study

  > Better seed management  

  > tau and lr schedulers

  > Automated GIF generation

* 1.3.0 - More environments

  > Added support for multiple environments
  
* 1.3.1 - More environments

  > Added support for `CartPole-v1`, `MountainCar-v0`, `Pendulum-v1`, and `BipedalWalker-v3`.

## How to Run This Project

**1. Clone the repository and install using the requirement.txt file:**

**2. Change config file as needed:**

Open `config.yaml` and set the `active_env` to the desired environment (e.g., `"LunarLander-v3"`, `"Acrobot-v1"`, `"CartPole-v1"`, etc.).

You can also configure the `ablation_study` section to choose the type of study:
* `study_type: 'component'`: The original buffer/target network ablation.
* `study_type: 'sweep'`: Sweep over a single hyperparameter.
* `study_type: 'dqn_variants'`: Compare DQN without Taget Network, DQN with Target Network , and Double DQN.

**3. Run ablation study:** runs the ablation, test the results, and generate gifs. 
```
python ablation_study.py
```
