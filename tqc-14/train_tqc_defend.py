#[Original Code:]https://github.com/SamsungLabs/tqc_pytorch

import numpy as np
import torch
import argparse
import os
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_agent.agent_builder_tqc import build_agent
from utils import ReplayBuffer, solve_hit_config_ik_null
from torch.utils.tensorboard.writer import SummaryWriter
from omegaconf import OmegaConf

from datetime import datetime
import copy
from reward import HitReward, DefendReward, PrepareReward

class train(AirHockeyChallengeWrapper):
    def __init__(self, env=None, custom_reward_function=DefendReward(), interpolation_order=3, **kwargs):
        # Load config file
        self.conf = OmegaConf.load('train_tqc_defend.yaml')
        env = self.conf.env
        # base env
        super().__init__(env, custom_reward_function,interpolation_order, **kwargs)
        # seed
        self.seed(self.conf.agent.seed)
        torch.manual_seed(self.conf.agent.seed)
        np.random.seed(self.conf.agent.seed)
        # env variables
        self.action_shape = 14
        # self.action_shape = 3
        self.observation_shape = self.env_info["rl_info"].observation_space.shape[0]
        # policy
        self.policy = build_agent(self.env_info)
        # action_space.high
        pos_max = self.env_info['robot']['joint_pos_limit'][1]
        vel_max = self.env_info['robot']['joint_vel_limit'][1] 
        max_ = np.stack([pos_max,vel_max])
        self.max_action  = max_.reshape(14,)
        # make dirs 
        self.make_dir()
        self.tensorboard = SummaryWriter(self.conf.agent.dump_dir + "/tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        # load model if defined
        if self.conf.agent.load_model!= "":
            policy_file = self.conf.agent.file_name if self.conf.agent.load_model == "default" else self.conf.agent .load_model
            print("loading model from file: ", policy_file)
            self.policy.load(self.conf.agent.dump_dir + f"/models/{policy_file}")
        
        self.replay_buffer = ReplayBuffer(self.observation_shape, self.action_shape)
    
    def _step(self,state,action):
        action = self.policy.action_scaleup(action)                                     # [-1,1] -> [-self.max_action,self.max_action]
        next_state, reward, done, info = self.step(action)
        

        return next_state, reward, done, info

    
    def make_dir(self):
        if not os.path.exists(self.conf.agent.dump_dir+"/results"):
            os.makedirs(self.conf.agent.dump_dir+"/results")

        if not os.path.exists(self.conf.agent.dump_dir+"/models"):
            os.makedirs(self.conf.agent.dump_dir+"/models")
    
   


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
    def eval_policy(self,t,eval_episodes=10):
        self.policy.actor.eval()
        for _ in range(eval_episodes):
            avg_reward = 0.
        
            state, done = self.reset(), False
            episode_timesteps=0
            while not done and episode_timesteps<100:
           

                action = self.policy.select_action(state)
                next_state, reward, done, info = self._step(state,action)
                # self.render()
                avg_reward += reward
                episode_timesteps+=1
                state = next_state
            self.tensorboard.add_scalar("eval_reward", avg_reward,t+_)
        self.policy.actor.train()


    def train_model(self):
        state, done = self.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
   
        for t in range(int(self.conf.agent.max_timesteps)):
            self.policy.actor.train()
            critic_loss = np.nan
            actor_loss = np.nan
            alpha_loss = np.nan

            episode_timesteps += 1
           
            # Select action randomly or according to policy
            if t < self.conf.agent.start_timesteps:
                action = np.random.uniform(-self.max_action,self.max_action,(self.action_shape,)).reshape(2,7)
            else:
                action = self.policy.select_action(state)
            # Perform action
            next_state, reward, done, _ = self._step(state,action) 
            # self.render()
            done_bool = float(done) if episode_timesteps < self.conf.agent.max_episode_steps else 0   ###MAX EPISODE STEPS
            # Store data in replay buffer
           
            self.replay_buffer.add(state, action.reshape(-1,), next_state, reward, done_bool)
            state = next_state
            episode_reward += reward

            # # Train agent after collecting sufficient data
            if t >= self.conf.agent.start_timesteps:
                actor_loss,critic_loss,alpha_loss=self.policy.train(self.replay_buffer, self.conf.agent.batch_size)

            if done or episode_timesteps > self.conf.agent.max_episode_steps: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward.sum():.3f}")
                # Reset environment
                if (actor_loss is not np.nan):
                    self.tensorboard.add_scalar("actor loss", actor_loss, t)
                if (critic_loss is not np.nan):
                    self.tensorboard.add_scalar("critic loss", critic_loss, t)
                if (alpha_loss is not np.nan):
                    self.tensorboard.add_scalar("alpha loss", alpha_loss, t)
                self.tensorboard.add_scalar("reward", episode_reward, t)
                state, done = self.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1 
                
            if (t + 1) % self.conf.agent.eval_freq == 0:
                self.eval_policy(t)
                self.policy.save(self.conf.agent.dump_dir + f"/models/{self.conf.agent.file_name}")

x = train()
x.train_model()
