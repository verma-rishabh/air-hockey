#[Original Code]:https://github.com/SamsungLabs/tqc_pytorch/tree/master/tqc
import numpy as np
import torch
import argparse
import os
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_agent.agent_builder_tqc_ik import build_agent
from utils import ReplayBuffer, solve_hit_config_ik_null
from torch.utils.tensorboard.writer import SummaryWriter
from omegaconf import OmegaConf
from air_hockey_challenge.utils.kinematics import inverse_kinematics, jacobian ,forward_kinematics
from datetime import datetime
import copy
from reward import HitReward, DefendReward, PrepareReward


class train(AirHockeyChallengeWrapper):
    def __init__(self, env=None, custom_reward_function=DefendReward(), interpolation_order=1, **kwargs):
        # Load config file
        self.episode_timesteps = 0
        self.conf = OmegaConf.load('train_tqc_defend.yaml')
        env = self.conf.env
        # base env
        super().__init__(env, custom_reward_function,interpolation_order, **kwargs)
        # seed
        self.seed(self.conf.agent.seed)
        torch.manual_seed(self.conf.agent.seed)
        np.random.seed(self.conf.agent.seed)
        # env variables
 
        self.action_shape = 2
        self.observation_shape = self.env_info["rl_info"].observation_space.shape[0]
        # policy
        self.policy = build_agent(self.env_info)

        self.min_action = np.array([0.65,-0.40, 0])
        self.max_action = np.array([1.32,0.40 , 1.5])
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
        action = self.policy.action_scaleup(action)
        # print(action)
        des_pos = np.array([action[0],action[1],0.1645])                                #'ee_desired_height': 0.1645

        x_ = [action[0],action[1]] 
        y = self.policy.get_ee_pose(state)[0][:2]


        _,x = inverse_kinematics(self.policy.robot_model, self.policy.robot_data,des_pos)
        #action_,_ = forward_kinematics(self.policy.robot_model, self.policy.robot_data, x)

        # _,x = solve_hit_config_ik_null(self.policy.robot_model,self.policy.robot_data, des_pos, des_v, self.policy.get_joint_pos(state))
        action = copy.deepcopy(x)
        next_state, reward, done, info = self.step(x)
        action_, _ = forward_kinematics(self.policy.robot_model, self.policy.robot_data, next_state[self.env_info['joint_pos_ids']])
        #self.tensorboard.add_scalars("action", {"action_step": action_[2]}, self.episode_timesteps)
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
            # print(_)
            state, done = self.reset(), False
            self.episode_timesteps=0
            while not done and self.episode_timesteps<100:
                # print("ep",episode_timesteps)

                action = self.policy.select_action(state)
                next_state, reward, done, info = self._step(state,action)
                # self.render()
                avg_reward += reward
                self.episode_timesteps+=1
                state = next_state
            self.tensorboard.add_scalar("eval_reward", avg_reward,t+_)
        self.policy.actor.train()

    def eval_plot_states(self,eval_episodes=10, episode_num=0):
        self.policy.actor.eval()
        t = int(self.conf.agent.max_timesteps)

        for _ in range(eval_episodes):
            avg_reward = 0.
            # print(_)
            state, done = self.reset(), False
            self.episode_timesteps=0
            while not done and self.episode_timesteps<500:
                # print("ep",episode_timesteps)

                action = self.policy.select_action(state)
                action_ = copy.deepcopy(action)
                next_state, reward, done, info = self._step(state,action)
                #self.render()
                print(self.episode_timesteps)
                if _ == episode_num:
                    q = next_state[self.env_info['joint_pos_ids']]
                    dq = next_state[self.env_info['joint_vel_ids']]

                    c_ee_ub = self.env_info['constraints'].get('ee_constr').z_ub
                    c_ee_lb = self.env_info['constraints'].get('ee_constr').z_lb

                    self.tensorboard.add_scalars("action", {"action_Z": self.policy.action_scaleup(action_)[2], "constraint_z_ub" : c_ee_ub,"constraint_z_ub" : c_ee_lb},
                                                self.episode_timesteps)

                avg_reward += reward
                self.episode_timesteps+=1
                state = next_state
            self.tensorboard.add_scalar("eval_reward", avg_reward,t+_)
        self.policy.actor.train()

    def train_model(self):
        state, done = self.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        # self.policy.actor.train()
        for t in range(int(self.conf.agent.max_timesteps)):
            self.policy.actor.train()
            critic_loss = np.nan
            actor_loss = np.nan
            alpha_loss = np.nan

            episode_timesteps += 1
           
            # Select action randomly or according to policy
            if t < self.conf.agent.start_timesteps:
                action = np.random.uniform(self.min_action,self.max_action,(self.action_shape,))
            else:
                action = self.policy.select_action(state)
            # Perform action
            next_state, reward, done, _ = self._step(state,action) 
            #self.render()
            done_bool = float(done) if episode_timesteps < self.conf.agent.max_episode_steps else 0   ###MAX EPISODE STEPS
            # Store data in replay buffer
            # print(action)
            self.replay_buffer.add(state, action.reshape(-1,), next_state, reward, done_bool)
            state = next_state
            episode_reward += reward

            # # Train agent after collecting sufficient data
            if t >= self.conf.agent.start_timesteps:
                actor_loss,critic_loss,alpha_loss=self.policy.train(self.replay_buffer, self.conf.agent.batch_size)

            if done or episode_timesteps > self.conf.agent.max_episode_steps: 
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {np.sum(episode_reward):.3f}")
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
                #self.eval_policy(t)
                self.policy.save(self.conf.agent.dump_dir + f"/models/{self.conf.agent.file_name}")

x = train()
x.train_model()
