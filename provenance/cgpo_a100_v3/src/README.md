# [ICML 2026] Sample-Efficient Diffusion-based Reinforcement Learning with Critic Guidance

Code release for **Sample-Efficient Diffusion-based Reinforcement Learning with Critic Guidance (ICML 2026)**.

[[paper]](http://arxiv.org/abs/2605.30056)

![](./asset/cgpo.png)

## Requirements
Installations of [PyTorch](https://pytorch.org/) and [MuJoCo](https://github.com/deepmind/mujoco) are needed. 
A suitable [conda](https://conda.io) environment named `cgpo` can be created and activated with:
```
conda create cgpo
conda activate cgpo
```
To get started, install the additionally required python packages into you environment.
```
pip install -r requirements.txt
```

## Running
Running experiments based our code could be quite easy, so below we use `HalfCheetah-v3` task as an example. 

```
python main.py --cuda cuda:0 --env HalfCheetah-v3 --weighted --aug --running_v --entropy_alpha 0.02 --num_steps 1000000 --seed 0
```

## Citation
If you find this repository useful in your research, please consider citing:

```
@inproceedings{
ding2026sampleefficient,
title={Sample-Efficient Diffusion-based Reinforcement Learning with Critic Guidance},
author={Shutong Ding, Zejia Zhong, Zhongyi Wang, Ke Hu, Bikang Pan, Jingya Wang, Ye Shi },
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
}
```
