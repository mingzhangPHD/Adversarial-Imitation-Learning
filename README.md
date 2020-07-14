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

[Wasserstein Distance guided Adversarial Imitation Learning (WDAIL) with Reward Shape Exploration](https://arxiv.org/abs/2006.03503).

# Citation

If you use this code and datasets for your research, please consider citing:

```
@article{zhang2020wasserstein,
  title={Wasserstein Distance guided Adversarial Imitation Learning with Reward Shape Exploration},
  author={Zhang, Ming and Wang, Yawei and Ma, Xiaoteng and Xia, Li and Yang, Jun and Li, Zhiheng and Li, Xiu},
  journal={arXiv preprint arXiv:2006.03503},
  year={2020}
}
```

# Contact
If you have any problem about our code, feel free to contact:

zhangming_0706@163.com

or describe your problem in issues
