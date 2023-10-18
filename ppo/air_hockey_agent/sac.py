import os
import random
import time
from distutils.util import strtobool
import sys
sys.path.append('/Users/zahrapadar/Desktop/DL-LAB/project/air_hockey_challenge_local_warmup/')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from air_hockey_challenge.framework.agent_base import AgentBase

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SoftQNetwork(nn.Module):

    def __init__(self, env_info):
        super().__init__()
        state_dim = env_info["rl_info"].observation_space.low.shape[0]
        action_dim = 2 * (env_info["rl_info"].action_space.low.shape[0])
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        # print(x.shape, a.shape)
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env_info):
        super().__init__()
        state_dim = env_info["rl_info"].observation_space.low.shape[0]
        action_dim = 2 * (env_info["rl_info"].action_space.low.shape[0])
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)
        _action_space = np.concatenate((env_info["robot"]["joint_pos_limit"], 
                                  env_info["robot"]["joint_vel_limit"]), axis=1)

        # max_action = float(envs.single_action_space.high[0])
        _max_action = _action_space[1,:]
        _min_action = _action_space[0,:]
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((_max_action - _min_action) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((_max_action + _min_action) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std
    
    def get_action(self, x):
        mean, log_std = self.forward(x.double())
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum()
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        if len(action.shape) < 3: 
            action = action.reshape(2,3)
        else:
            action = action.reshape((action.shape[0],2,3))
        return action.detach().numpy(), log_prob, mean
    

class SAC_Agent(AgentBase, nn.Module):

    def __init__(self, env_info, agent_id=1):
        super().__init__(env_info, agent_id)
        nn.Module.__init__(self)
        # self.env = env
        self.state_dim = env_info["rl_info"].observation_space.low.shape[0]
        self.action_dim = 2 * env_info["rl_info"].action_space.low.shape[0]
        self.action_dim_ = 2 * env_info["rl_info"].action_space.low.shape[0]

        self.actor = Actor( env_info)
        self.qf1 = SoftQNetwork( env_info)
        self.qf2 = SoftQNetwork( env_info) 
        self.qf1_target = SoftQNetwork( env_info) 
        self.qf2_target = SoftQNetwork( env_info) 
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

    def get_value(self, x):
        return self.critic(x)
    
    def draw_action(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        action, _, _ =  self.actor.get_action(x)
        return action
    
    def save(self, filename):
        # print(self.critic.state_dict())
        torch.save(self.qf1.state_dict(), filename + "_qf1")
        torch.save(self.qf2.state_dict(), filename + "_qf2")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.qf1_target.state_dict(), filename + "_qf1_target")
        torch.save(self.qf2_target.state_dict(), filename + "_qf2_target")


    def load(self, filename):
        print("loading agent: ....")

        state_dict_am = torch.load(filename + "_qf1")
        self.qf1.load_state_dict(state_dict_am)

        state_dict_am = torch.load(filename + "_qf2")
        self.qf2.load_state_dict(state_dict_am)

        state_dict_am = torch.load(filename + "_actor")
        self.actor.load_state_dict(state_dict_am)

        state_dict_am = torch.load(filename + "_qf1_target")
        self.qf1_target.load_state_dict(state_dict_am)

        state_dict_am = torch.load(filename + "_qf2_target")
        self.qf2_target.load_state_dict(state_dict_am)


    def reset(self):
        pass
        # self.env.reset()
