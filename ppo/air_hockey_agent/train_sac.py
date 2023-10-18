# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import argparse
import os
import random
import sys
import time
from distutils.util import strtobool
import copy
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from stable_baselines3.common.buffers import ReplayBuffer
from utils import ReplayBuffer

from torch.utils.tensorboard import SummaryWriter
from air_hockey_challenge.framework.agent_base import AgentBase
from air_hockey_challenge.framework import AirHockeyChallengeWrapper
from air_hockey_agent.sac import SAC_Agent, SoftQNetwork, Actor

# from air_hockey_agent.agent_builder import build_agent_ppo
from torch.utils.tensorboard import SummaryWriter
sys.path.append('/Users/zahrapadar/Desktop/DL-LAB/project/air_hockey_challenge_local_warmup/')


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--env-id", type=str, default="3dof-hit",
        help="the id of the environment")
    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=10000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=5e3,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-3,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument("--alpha", type=float, default=0.2,
            help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    return args


def make_env(env_id, seed, custom_reward_function=None):
    env = AirHockeyChallengeWrapper(env_id, custom_reward_function=custom_reward_function, interpolation_order=3)
    env.seed = seed
    env.env_info["rl_info"].observation_space.seed = seed

    return env

def reward_mushroomrl(env, state, action, next_state):
# print("calculating rewardd")
    r = 0
    mod_next_state = next_state                            # changing frame of puck pos (wrt origin)
    mod_next_state[:3]  = mod_next_state[:3] - [1.51,0,0.1]
    absorbing = env.base_env.is_absorbing(mod_next_state)
    puck_pos, puck_vel = env.base_env.get_puck(mod_next_state)                     # extracts from obs therefore robot frame
    # print("puck_velocity", puck_vel)                    # extracts from obs therefore robot frame

    ###################################################
    goal = np.array([0.974, 0])
    effective_width = 0.519 - 0.03165

    # Calculate bounce point by assuming incoming angle = outgoing angle
    w = (abs(puck_pos[1]) * goal[0] + goal[1] * puck_pos[0] - effective_width * puck_pos[
        0] - effective_width *
            goal[0]) / (abs(puck_pos[1]) + goal[1] - 2 * effective_width)


    side_point = np.array([w, np.copysign(effective_width, puck_pos[1])])
    #print("side_point",side_point)

    vec_puck_side = (side_point - puck_pos[:2]) / np.linalg.norm(side_point - puck_pos[:2])
    vec_puck_goal = (goal - puck_pos[:2]) / np.linalg.norm(goal - puck_pos[:2])
    has_hit = env.base_env._check_collision("puck", "robot_1/ee")

    
    ###################################################
    
    

    # If puck is out of bounds
    if absorbing:
        # If puck is in the opponent goal
        if (puck_pos[0] - env.env_info['table']['length'] / 2) > 0 and \
                (np.abs(puck_pos[1]) - env.env_info['table']['goal_width']) < 0:
                # print("puck_pos",puck_pos,"absorbing",absorbing)
            r = 200

    else:
        if not has_hit:
            ee_pos = env.base_env.get_ee()[0]                                     # tO check
            # print(ee_pos,self.policy.get_ee_pose(next_state)[0] - [1.51,0,0.1])

            dist_ee_puck = np.linalg.norm(puck_pos[:2] - ee_pos[:2])                # changing to 2D plane because used to normalise 2D vector

            vec_ee_puck = (puck_pos[:2] - ee_pos[:2]) / dist_ee_puck

            cos_ang_side = np.clip(vec_puck_side @ vec_ee_puck, 0, 1)

            # Reward if vec_ee_puck and vec_puck_goal have the same direction
            cos_ang_goal = np.clip(vec_puck_goal @ vec_ee_puck, 0, 1)
            cos_ang = np.max([cos_ang_goal, cos_ang_side])

            r = np.exp(-8 * (dist_ee_puck - 0.08)) * cos_ang ** 2
        else:
            r_hit = 0.25 + min([1, (0.25 * puck_vel[0] ** 4)])

            r_goal = 0
            if puck_pos[0] > 0.7:
                sig = 0.1
                r_goal = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((puck_pos[1] - 0) / sig, 2.) / 2)

            r = 2 * r_hit + 10 * r_goal

    r -= 1e-3 * np.linalg.norm(action)
    
    des_z = env.env_info['robot']['ee_desired_height']
    tolerance = 0.02

    # if abs(self.policy.get_ee_pose(next_state)[0][1])>0.519:         # should replace with env variables some day
        #     r -=1 
        # if (self.policy.get_ee_pose(next_state)[0][0])<0.536:
        #     r -=1 
        # if (self.policy.get_ee_pose(next_state)[0][2]-0.1)<des_z-tolerance*10 or (self.policy.get_ee_pose(next_state)[0][2]-0.1)>des_z+tolerance*10:
        #     r -=1
    return r    


if __name__ == "__main__":
    args = parse_args()
    timestamp = time.time()
    formatted_time = time.strftime("%d-%m-%Y %H:%M", time.localtime(timestamp))

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{formatted_time}"

    writer = SummaryWriter(f"runs/sac/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = make_env(args.env_id, args.seed, custom_reward_function=None)

    env_info = envs.env_info

    state_dim = env_info["rl_info"].observation_space.low.shape[0]
    state_dim_ = env_info["rl_info"].observation_space.low.shape

    action_dim = 2 * (env_info["rl_info"].action_space.low.shape[0])
    action_dim_ = (action_dim,)
    state, done = envs.reset(), False #initial_state

    action_space = np.concatenate((env_info["robot"]["joint_pos_limit"], 
                                  env_info["robot"]["joint_vel_limit"]), axis=1)

    # max_action = float(envs.single_action_space.high[0])
    max_action = action_space[1,:]
    min_action = action_space[0,:]

    agent = SAC_Agent(env_info)

    actor = agent.actor.to(device)
    qf1 = agent.qf1.to(device)
    qf2 = agent.qf2.to(device)
    qf1_target = agent.qf1_target.to(device)
    qf2_target = agent.qf2_target.to(device)

    # qf1_target.load_state_dict(qf1.state_dict())
    # qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(2 * (envs.env_info["rl_info"].action_space.low.shape[0])).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    # envs.single_observation_space.dtype = np.float32
    # env_info["rl_info"].action_space.low.dtype = np.float32
    # env_info["rl_info"].action_space.high.dtype = np.float32

    rb = ReplayBuffer(state_dim, action_dim, max_size=args.buffer_size)

    # rb = ReplayBuffer(
    #     args.buffer_size,
    #     envs.single_observation_space,
    #     envs.single_action_space,
    #     device,
    #     handle_timeout_termination=True,
    # )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = torch.Tensor([random.uniform(min_action[i], max_action[i]) for i in range(action_dim)]).reshape(2,3)
            # actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))

        # if len(actions.shape) < 3 : 
        #     actions=actions.reshape(2,3)
        # else:
        #     actions = actions.reshape((actions.shape[0],2,3))
            
        actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)
        dones = 1 if dones else 0

        # # TRY NOT TO MODIFY: record rewards for plotting purposes
        # for info in infos:
        #     if "episode" in info.keys():
        #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
        #         writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
        #         writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
        #         break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        # real_next_obs = next_obs.copy()
        
        # if dones:
        #     real_next_obs = terminal_state
        # rb.add(obs, real_next_obs, actions, rewards, dones, infos)
        if len(actions.shape) < 3:
            actions = actions.flatten()
        else:
            actions = actions.reshape(actions.shape[0],-1)

        rb.add(state,actions,next_obs,rewards,done)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        
        if not done:
            obs = next_obs
        else: 
            obs = envs.reset()
    
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data[2])
                qf1_next_target = qf1_target(data[2], next_state_actions)
                qf2_next_target = qf2_target(data[2], next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data[3].flatten() + (1 - data[4].flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data[0], data[1]).view(-1)
            qf2_a_values = qf2(data[0], data[1]).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data[0])
                    qf1_pi = qf1(data[0], pi)
                    qf2_pi = qf2(data[0], pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi).view(-1)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data[0])
                        alpha_loss = (-log_alpha * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                print(global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    agent.save(f"./models/sac_agent")
    envs.stop()
    writer.close()