# Version 1.2.0 discussion

## Version features
**changes**:

Added Double DQN logic.

**This version include:**

* Double DQN scheme
* 3 Fully Connected Linear Layers (State Space: 8 -> 32 -> 32 -> Action Space: 4)
* ReLU Activation functions
* $\epsilon$-greedy policy with decay.
* Mean Squared Error (MSE) Loss with Adam Optimizer.
* Experience Replay Buffer
* Fixed Q-Targets (Local+Target Networks)
* Rewards are defined by the environment

## Results:

* best landing recorded:
![alt text](../gif/1-2-0_DDQN_HiddenSize_64-64_best-episode-0.gif)

* It does manage to learn how to land the ship and the training proccess is still pretty noisy:
![alt text](../images/ver_1-2-1_training.png)

* I did encountered something some may call the "Catastrophic Forgetting" problem: 

We may forget past experience prematurly. I will try to address it in the future

* I did see the best agent fail miserably occasionally!

## Future ideas:

* make experiment infrastructure. 
* experiment with NN architecture, add hidden layers, test 'bottleneck' architectures.
* play with replay buffer length to see if it helps with the catastrophic forgetting problem.