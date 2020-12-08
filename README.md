# Adversarial-Imitation-Learning
# 模仿学习 对抗学习 Mujoco

This repository is for improving the GAIL which is called WDAIL.

People should download the [Mojoco demostrations](https://data.mendeley.com/datasets/w7m95wwrb5/1) and put is in WDAIL\data,
or you can get the \data\baseline from the [openai\baseline](https://github.com/openai/baselines) 
and the \data\ikostrikov frome the [ikostrikov/pytorch-a2c-ppo-acktr-gail](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail),
or you can produce the demonstrations using our [PPO_pytorch](https://github.com/mingzhangPHD/RL-trip-pytorch)

Then, you can run wdail_ant.py, wdail_halfcheetah...in ppo_wdail and ppo_wdail_BL.

Finally, you can get the results.

The paper is as follow:

[Wasserstein Distance guided Adversarial Imitation Learning (WDAIL) with Reward Shape Exploration](https://ieeexplore.ieee.org/document/9275169).

# Citation

If you use this code and datasets for your research, please consider citing:

```
@INPROCEEDINGS{9275169,
  author={M. {Zhang} and Y. {Wang} and X. {Ma} and L. {Xia} and J. {Yang} and Z. {Li} and X. {Li}},
  booktitle={2020 IEEE 9th Data Driven Control and Learning Systems Conference (DDCLS)}, 
  title={Wasserstein Distance guided Adversarial Imitation Learning with Reward Shape Exploration}, 
  year={2020},
  volume={},
  number={},
  pages={1165-1170},
  abstract={The generative adversarial imitation learning (GAIL) has provided an adversarial learning framework for imitating expert policy from demonstrations in high-dimensional continuous tasks. However, almost all GAIL and its extensions only design a kind of reward function of logarithmic form in the adversarial training strategy with the Jensen-Shannon (JS) divergence for all complex environments. The fixed logarithmic type of reward function may be difficult to solve all complex tasks, and the vanishing gradients problem caused by the JS divergence will harm the adversarial learning process. In this paper, we propose a new algorithm named Wasserstein Distance guided Adversarial Imitation Learning (WDAIL) for promoting the performance of imitation learning (IL). There are three improvements in our method: (a) introducing the Wasserstein distance to obtain more appropriate measure in adversarial training process, (b) using proximal policy optimization (PPO) in the reinforcement learning stage which is much simpler to implement and makes the algorithm more efficient, and (c) exploring different reward function shapes to suit different tasks for improving the performance. The experiment results show that the learning procedure remains remarkably stable, and achieves significant performance in the complex continuous control tasks of MuJoCo1.},
  keywords={Shape;Task analysis;Optimization;Generative adversarial networks;Training;Q measurement;Reinforcement learning;Generative Adversarial Imitation Learning;Proximal Policy Optimization;Wasserstein Distance;Reward Shaping},
  doi={10.1109/DDCLS49620.2020.9275169},
  ISSN={},
  month={Nov},}
```

# Contact
If you have any problem about our code, feel free to contact:

zhangming_0706@163.com

or describe your problem in issues
