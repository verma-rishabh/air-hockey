import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from air_hockey_challenge.framework.agent_base import AgentBase
from omegaconf import OmegaConf
from torch.distributions import Distribution, Normal
from scipy import sparse
from casadi import SX, sin, Function, inf,vertcat,nlpsol,qpsol,sumsqr
from air_hockey_challenge.utils.kinematics import inverse_kinematics, jacobian, forward_kinematics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#[Original Code:]https://github.com/SamsungLabs/tqc_pytorch/tree/master/tqc


LOG_STD_MIN_MAX = (-20, 2)

class Mlp(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size
    ):
        super().__init__()
        # TODO: initialization
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            self.add_module(f'fc{i}', fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = F.relu(fc(h))
        output = self.last_fc(h)
        return output


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_quantiles, n_nets):
        super().__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets
        for i in range(n_nets):
            net = Mlp(state_dim + action_dim, [128, 128], n_quantiles)
            self.add_module(f'qf{i}', net)
            self.nets.append(net)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim,env_info,agent_id=1):
        super().__init__()
        self.action_dim = action_dim
        self.net = Mlp(state_dim, [128,128], 2 * action_dim)

    def forward(self, obs):
        mean, log_std = self.net(obs).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(*LOG_STD_MIN_MAX)

        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TanhNormal(mean, std)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:  # deterministic eval without log_prob computation
            action = torch.tanh(mean)
            log_prob = None
        return action, log_prob

    

class TQC_agent(AgentBase):
    def __init__(
        self,
        env_info,
        agent_id
    ):
        super().__init__(env_info, agent_id)
        conf = OmegaConf.load('train_tqc.yaml')

        state_dim = env_info["rl_info"].observation_space.shape[0]
        #action_dim = env_info["rl_info"].action_space.shape[0]
        action_dim = 2
        self.min_action = np.array([0.65,-0.40])
        self.max_action = np.array([1.32,0.40])
        state_max = np.array(env_info['rl_info'].observation_space.high,dtype=np.float32)
        self.state_max = torch.from_numpy(state_max).to(device)
        max_action = np.array([1.5,0.5,5],dtype=np.float32)
        max_action = torch.from_numpy(max_action).to(device)
   
        
        discount = conf.agent.discount
        tau=conf.agent.tau
        
        discount = conf.agent.discount
        tau = conf.agent.tau
        target_entropy = -np.prod(action_dim).item()
        n_nets = conf.agent.n_nets
        n_quantiles = conf.agent.n_quantiles
        top_quantiles_to_drop = conf.agent.top_quantiles_to_drop_per_net * n_nets
        

        self.actor = Actor(state_dim, action_dim,env_info).to(device)
        self.critic = Critic(state_dim, action_dim, n_quantiles,n_nets).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.log_alpha = torch.zeros((1,), requires_grad=True, device=device)

        # TODO: check hyperparams
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-3)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)

        self.discount = discount
        self.tau = tau
        self.top_quantiles_to_drop = top_quantiles_to_drop
        self.target_entropy = target_entropy

        self.quantiles_total = n_quantiles * n_nets

        self.total_it = 0

    def action_scaleup(self, action):
        low = self.min_action
        high = self.max_action
        a = np.zeros_like(low) -1.0
        b = np.zeros_like(low) +1.0
        action = low + (high - low)*((action - a)/(b - a))
        action = np.clip(action, low, high)
        return action



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
        N = 5
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
            ubw += [1.5,.5,0.165]
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
        if(not _):
            reward = -10
            done = 1
            print("failed")
            return state, reward, done, {}
    # print(q)
        next_q = q_0 + q*self.env_info['dt']
        # # _,x = inverse_kinematics(self.policy.robot_model, self.policy.robot_data,des_pos)
        # _,x = self._solve_aqp(des_pos,self.policy.robot_data,self.policy.joint_anchor_pos)
        # # _,x = solve_hit_config_ik_null(self.policy.robot_model,self.policy.robot_data, des_pos, des_v, self.policy.get_joint_pos(state))
        # q_cur = x * self.env_info['dt']
        action = copy.deepcopy(next_q)
        next_state, reward, done, info = self.step(action)


        return next_state, reward, done, info

    def draw_action(self, state):
        norm_state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.action_scaleup(self.actor(norm_state)[0][0].cpu().detach().numpy())
        des_pos = np.array([action[0],action[1],0.1645])                                #'ee_desired_height': 0.1645

        # x_ = [action[0],action[1]] 
        # y = self.policy.get_ee_pose(state)[0][:2]
        # des_v = action[2]*(x_-y)/(np.linalg.norm(x_-y)+1e-8)
        # des_v = np.concatenate((des_v,[0])) 
        q_0 = state[6:13]
        jac = jacobian(self.robot_model, self.robot_data,q_0)[:3, :7]
        x0 = list(forward_kinematics(self.robot_model, self.robot_data, q_0)[0])
        q,_ = self.solve_casadi(x0,des_pos,jac)
        if(not _):
            print("failed")
        
        next_q = q_0 + q*self.env_info['dt']
        return next_q

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # return self.action_scaleup(self.actor(state)[0][0].cpu().detach().numpy())
        return self.actor(state)[0][0].cpu().detach().numpy()


    def train(self, replay_buffer, batch_size=256):

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        # action = self.action_scaledown(action.cpu().detach().numpy())

        alpha = torch.exp(self.log_alpha)

        # --- Q loss ---
        with torch.no_grad():
            # get policy action
            new_next_action, next_log_pi = self.actor(next_state)

            # compute and cut quantiles at the next state
            next_z = self.critic_target(next_state, new_next_action)  # batch x nets x quantiles
            sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
            sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]

            # compute target
            target = reward + not_done * self.discount * (sorted_z_part - alpha * next_log_pi)

        cur_z = self.critic(state, action)
        critic_loss = self.quantile_huber_loss_f(cur_z, target)

        # --- Policy and alpha loss ---
        new_action, log_pi = self.actor(state)
        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()
        actor_loss = (alpha * log_pi - self.critic(state, new_action).mean(2).mean(1, keepdim=True)).mean()

        # --- Update ---
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.total_it += 1

        return actor_loss.item(), critic_loss.item(), alpha_loss.item()

    def save(self, filename):
        filename = str(filename)
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_target.state_dict(), filename + "_critic_target")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
        torch.save(self.log_alpha, filename + '_log_alpha')
        torch.save(self.alpha_optimizer.state_dict(), filename + "_alpha_optimizer")

    def load(self, filename):
        filename = str(filename)
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.log_alpha = torch.load(filename + '_log_alpha')
        self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer"))

    def quantile_huber_loss_f(self,quantiles, samples):
        pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
        abs_pairwise_delta = torch.abs(pairwise_delta)
        huber_loss = torch.where(abs_pairwise_delta > 1,
                                abs_pairwise_delta - 0.5,
                                pairwise_delta ** 2 * 0.5)

        n_quantiles = quantiles.shape[2]
        tau = torch.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
        loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
        return loss
    
    def reset(self):
        pass
        # return super().reset()
    
class TanhNormal(Distribution):
    def __init__(self, normal_mean, normal_std):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std
        self.standard_normal = Normal(torch.zeros_like(self.normal_mean, device=device),
                                      torch.ones_like(self.normal_std, device=device))
        self.normal = Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = 2 * np.log(2) + F.logsigmoid(2 * pre_tanh) + F.logsigmoid(-2 * pre_tanh)
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh
