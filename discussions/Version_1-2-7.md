# Version 1.2.0 discussion

## Version features
**changes**:

* Added ablation study.
* Fixed seed issue
* Automated gif generation
* Schedulers - for tau and lr
* QoL improvement - folder reordered

**This version include:**

* Double DQN scheme
* Ablation study

## Results:

* Ablation study results for experience replay size:
![alt text](../gif/ReplayBuffer_sweep_EnvSeed4_ModelSeed(1).gif)

* Score 
![alt text](../images/buffer_size_ablation_1-2-7/scores_comparison.png)

and its test scores distribution:
![alt text](../images/buffer_size_ablation_1-2-7/test_scores_violin.png)


## Future ideas:

* Add a target network ablation study
* Add another environment: maybe Acrobot 