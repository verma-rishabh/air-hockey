#[Original Code]:https://github.com/SamsungLabs/tqc_pytorch/tree/master/tqc
import numpy as np
import torch
import argparse
import os
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_agent.agent_builder_tqc import build_agent
from utils import ReplayBuffer, solve_hit_config_ik_null
from torch.utils.tensorboard.writer import SummaryWriter
from omegaconf import OmegaConf
from air_hockey_challenge.utils.kinematics import inverse_kinematics, jacobian, forward_kinematics
from datetime import datetime
import copy
from reward import HitReward, DefendReward, PrepareReward
from scipy import sparse
from casadi import SX, sin, Function, inf,vertcat,nlpsol,qpsol,sumsqr

class train(AirHockeyChallengeWrapper):
    def __init__(self, env=None, custom_reward_function=HitReward(), interpolation_order=1, **kwargs):
        # Load config file
        self.conf = OmegaConf.load('train_tqc_hit.yaml')
        env = self.conf.env
        # base env
        super().__init__(env, custom_reward_function,interpolation_order, **kwargs)
        # seed
        self.seed(self.conf.agent.seed)
        torch.manual_seed(self.conf.agent.seed)
        np.random.seed(self.conf.agent.seed)
        # env variables
 
        self.action_shape = 3
        self.observation_shape = self.env_info["rl_info"].observation_space.shape[0]
        # policy
        self.policy = build_agent(self.env_info)

        self.min_action = np.array([0.65,-0.40,0])
        self.max_action = np.array([1.32,0.40,1.5])
        # make dirs 
        self.make_dir()
        self.tensorboard = SummaryWriter(self.conf.agent.dump_dir + "/tensorboard/" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        # load model if defined
        if self.conf.agent.load_model!= "":
            policy_file = self.conf.agent.file_name if self.conf.agent.load_model == "default" else self.conf.agent .load_model
            print("loading model from file: ", policy_file)
            self.policy.load(self.conf.agent.dump_dir + f"/models/{policy_file}")
        else:
            self.policy.load(self.conf.agent.dump_dir + f"/models/"+"tqc-hit_7dof-hit_1")
        self.replay_buffer = ReplayBuffer(self.observation_shape, self.action_shape)
    
    def integrate_RK4(self,s_expr, a_expr, sdot_expr, dt, N_steps=1):
        '''RK4 integrator.

        s_expr, a_expr: casadi expression that have been used to define the dynamics sdot_expr
        sdot_expr:      casadi expr defining the rhs of the ode
        dt:             integration interval
        N_steps:        number of integration steps per integration interval, default:1
        '''
        dt = self.env_info['dt']
        h = dt/N_steps

        s_end = s_expr

        sdot_fun = Function('xdot', [s_expr, a_expr], [sdot_expr])

        for _ in range(N_steps):

        # FILL IN YOUR CODE HERE
            v_1 = sdot_fun(s_end, a_expr)
            v_2 = sdot_fun(s_end + 0.5 * h * v_1, a_expr)
            v_3 = sdot_fun(s_end + 0.5 * h * v_2, a_expr)
            v_4 = sdot_fun(s_end + v_3 * h, a_expr)
            s_end += (1/6) * (v_1 + 2 * v_2 + 2 * v_3 + v_4) * h

        F_expr = s_end

        return F_expr
    
    def solve_casadi(self,x0_bar,x_des,jac):
        # continuous model dynamics
        n_s = 3  # number of states
        n_a = 7  # number of actions

        x = SX.sym('x')
        y = SX.sym('y')
        z = SX.sym('z')

        omega = SX.sym('omega',7)

        s = vertcat(x,y,z)
        # q_0 = policy.robot_data.qpos.copy()
        # jac = jacobian(policy.robot_model, policy.robot_data,q_0)[:3, :7]
        s_dot = vertcat(jac @ omega)
        # Define number of steps in the control horizon and discretization step
        # print(s_dot)
        N = 20
        delta_t = 1/50
        # Define RK4 integrator function and initial state x0_bar
        F_rk4 = Function("F_rk4", [s, omega], [self.integrate_RK4(s, omega, s_dot, delta_t)])
        # x0_bar = [-.5, .5,.165]

        # Define the weighting matrix for the cost function
        Q = np.eye(n_s)
        R = np.eye(n_a)

            # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = SX.sym('X0', 3)
        w += [Xk]
        lbw += x0_bar    # set initial state
        ubw += x0_bar    # set initial state
        w0 += x0_bar     # set initial state

        # Formulate the NLP
        for k in range(N):
            # New NLP variable for the control
            Uk = SX.sym('U_' + str(k),7)
            w   += [Uk]
            lbw += [-1.48352986, -1.48352986, -1.74532925, -1.30899694, -2.26892803,
            -2.35619449, -2.35619449]
            ubw += [1.48352986, 1.48352986, 1.74532925, 1.30899694, 2.26892803,
            2.35619449, 2.35619449]
            w0  += [0,0,0,0,0,0,0]

            # Integrate till the end of the interval
            Xk_end = F_rk4(Xk, Uk)
            # J = J + delta_t *(sumsqr((Xk-x_des).T @ Q )+ sumsqr(R@Uk)) # Complete with the stage cost
            J = J + (sumsqr((Xk-x_des))) # Complete with the stage cost

            # New NLP variable for state at end of interval
            Xk = SX.sym(f'X_{k+1}', 3)
            w += [Xk]
            lbw += [.5,-.5,0.165]
            ubw += [1.5,.5,0.175]
            w0 += [0, 0,0]

            # Add equality constraint to "close the gap" for multiple shooting
            g   += [Xk_end-Xk]
            lbg += [0, 0,0]
            ubg += [0, 0,0]
        J = J + sumsqr((Xk-x_des)) # Complete with the terminal cost (NOTE it should be weighted by delta_t)

        # Create an NLP solver
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob,{'ipopt':{'print_level':0}, 'print_time': False})

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()
        # return np.array([w_opt[3::10],w_opt[4::10],w_opt[5::10],w_opt[6::10],w_opt[7::10],w_opt[8::10],w_opt[9::10]])
        
        return np.array(w_opt[3:10]),solver.stats()['success']
        # return solver
    

    
    def _step(self,state,action):
        action = self.policy.action_scaleup(action)
        # print(action)
        des_pos = np.array([action[0],action[1],0.1645])                                #'ee_desired_height': 0.1645

        # x_ = [action[0],action[1]] 
        # y = self.policy.get_ee_pose(state)[0][:2]
        # des_v = action[2]*(x_-y)/(np.linalg.norm(x_-y)+1e-8)
        # des_v = np.concatenate((des_v,[0])) 
        q_0 = state[6:13]
        jac = jacobian(self.policy.robot_model, self.policy.robot_data,q_0)[:3, :7]
        x0 = list(forward_kinematics(self.policy.robot_model, self.policy.robot_data, q_0)[0])
        q,_ = self.solve_casadi(x0,des_pos,jac)
        # if(not _):
        #     reward = -1
        #     done = 0
        #     return state, reward, done, {}
    # print(q)
        next_q = q_0 + q*0.02
        # # _,x = inverse_kinematics(self.policy.robot_model, self.policy.robot_data,des_pos)
        # _,x = self._solve_aqp(des_pos,self.policy.robot_data,self.policy.joint_anchor_pos)
        # # _,x = solve_hit_config_ik_null(self.policy.robot_model,self.policy.robot_data, des_pos, des_v, self.policy.get_joint_pos(state))
        # q_cur = x * self.env_info['dt']
        action = copy.deepcopy(next_q)
        next_state, reward, done, info = self.step(action)


        return next_state, reward, done, info

    # def _step(self,state,action):
    #     action = self.policy.action_scaleup(action)
    #     # print(action)
    #     des_pos = np.array([action[0],action[1],0.1645])                                #'ee_desired_height': 0.1645

    #     x_ = [action[0],action[1]] 
    #     y = self.policy.get_ee_pose(state)[0][:2]
    #     des_v = action[2]*(x_-y)/(np.linalg.norm(x_-y)+1e-8)
    #     des_v = np.concatenate((des_v,[0])) 
    #     _,x = inverse_kinematics(self.policy.robot_model, self.policy.robot_data,des_pos)
    #     # _,x = solve_hit_config_ik_null(self.policy.robot_model,self.policy.robot_data, des_pos, des_v, self.policy.get_joint_pos(state))
    #     action = copy.deepcopy(x)
    #     next_state, reward, done, info = self.step(x)


    #     return next_state, reward, done, info

    
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
            episode_timesteps=0
            while not done and episode_timesteps<100:
                # print("ep",episode_timesteps)

                action = self.policy.select_action(state)
                next_state, reward, done, info = self._step(state,action)
                self.render()
                avg_reward += reward
                episode_timesteps+=1
                state = next_state
            self.tensorboard.add_scalar("eval_reward", avg_reward,t+_)
        self.policy.actor.train()

    def eval_plot_states(self,eval_episodes=10, episode_num=0):
        self.policy.actor.eval()
        for t in range(int(self.conf.agent.max_timesteps)):

            for _ in range(eval_episodes):
                avg_reward = 0.
                # print(_)
                state, done = self.reset(), False
                episode_timesteps=0
                while not done and episode_timesteps<100:
                    # print("ep",episode_timesteps)

                    action = self.policy.select_action(state)
                    next_state, reward, done, info = self._step(state,action)
                    #self.render()
                    if _ == episode_num:
                        self.tensorboard.add_scalar("action", action[2],episode_timesteps)
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
        # self.policy.actor.train()
        for t in range(int(self.conf.agent.max_timesteps)):
            self.eval_policy(t)

            # self.policy.actor.train()
            # critic_loss = np.nan
            # actor_loss = np.nan
            # alpha_loss = np.nan

            # episode_timesteps += 1
           
            # # Select action randomly or according to policy
            # if t < self.conf.agent.start_timesteps:
            #     action = np.random.uniform(self.min_action,self.max_action,(self.action_shape,))
            # else:
            #     action = self.policy.select_action(state)
            # # Perform action
            # next_state, reward, done, _ = self._step(state,action) 
            # # self.render()
            # done_bool = float(done) if episode_timesteps < self.conf.agent.max_episode_steps else 0   ###MAX EPISODE STEPS
            # # Store data in replay buffer
            # # print(action)
            # self.replay_buffer.add(state, action.reshape(-1,), next_state, reward, done_bool)
            # state = next_state
            # episode_reward += reward

            # # # Train agent after collecting sufficient data
            # if t >= self.conf.agent.start_timesteps:
            #     actor_loss,critic_loss,alpha_loss=self.policy.train(self.replay_buffer, self.conf.agent.batch_size)

            # if done or episode_timesteps > self.conf.agent.max_episode_steps: 
            #     # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            #     print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward.sum():.3f}")
            #     # Reset environment
            #     if (actor_loss is not np.nan):
            #         self.tensorboard.add_scalar("actor loss", actor_loss, t)
            #     if (critic_loss is not np.nan):
            #         self.tensorboard.add_scalar("critic loss", critic_loss, t)
            #     if (alpha_loss is not np.nan):
            #         self.tensorboard.add_scalar("alpha loss", alpha_loss, t)
            #     self.tensorboard.add_scalar("reward", episode_reward, t)
            #     state, done = self.reset(), False
            #     episode_reward = 0
            #     episode_timesteps = 0
            #     episode_num += 1 
                
            # if (t + 1) % self.conf.agent.eval_freq == 0:
            #     self.eval_policy(t)
            #     self.policy.save(self.conf.agent.dump_dir + f"/models/{self.conf.agent.file_name}")

x = train()
x.eval_plot_states()
