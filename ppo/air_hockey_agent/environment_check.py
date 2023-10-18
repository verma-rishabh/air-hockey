
# %%
import argparse
import os
import random
import time
from distutils.util import strtobool
import sys
sys.path.append('/Users/zahrapadar/Desktop/DL-LAB/project/air_hockey_challenge_local_warmup/')

# import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.framework import AirHockeyChallengeWrapper

# %%
env = AirHockeyChallengeWrapper("3dof-hit", custom_reward_function=None, interpolation_order=3)
# print(en.step(action=))/
# print(en.env_info["rl_info"].observation_space.low)
print(env.reset())
obs, reward, done, info = env.step(np.zeros((2,3)))
print("INfo", info.keys())
state_dim = env.env_info["rl_info"].observation_space.low.shape
action_dim = 2 * env.env_info["rl_info"].action_space.low.shape
action_dim = 6
env_info = env.env_info
pos_max = env.env_info['robot']['joint_pos_limit'][1]
vel_max = env.env_info['robot']['joint_vel_limit'][1] 
max_ = np.stack([pos_max,vel_max],dtype=np.float32)
max_action  = max_.reshape(6,)
max_action = torch.from_numpy(max_action)
raise
print('initial state shape', env.reset().shape)
# %%