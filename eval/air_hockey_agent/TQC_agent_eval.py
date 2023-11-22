import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from air_hockey_challenge.framework.agent_base import AgentBase
# from omegaconf import OmegaConf
from torch.distributions import Distribution, Normal
from utils import solve_hit_config_ik_null
from air_hockey_challenge.utils.kinematics import inverse_kinematics, jacobian
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


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
        # conf = OmegaConf.load('train_tqc.yaml')

        state_dim = env_info["rl_info"].observation_space.shape[0]
        #action_dim = env_info["rl_info"].action_space.shape[0]
        action_dim = 2
        self.min_action = np.array([0.65,-0.40])
        self.max_action = np.array([1.32,0.40])
        state_max = np.array(env_info['rl_info'].observation_space.high,dtype=np.float32)
        self.state_max = torch.from_numpy(state_max).to(device)
        # max_action = np.array([1.5,0.5,5],dtype=np.float32)
        # max_action = torch.from_numpy(max_action).to(device)
        # pos_max = self.env_info['robot']['joint_pos_limit'][1]
        # vel_max = self.env_info['robot']['joint_vel_limit'][1] 
        # max_ = np.stack([pos_max,vel_max])
        # self.final_max_action  = max_.reshape(14,)
        
        discount = 0.99
        tau=0.005
        target_entropy = -np.prod(action_dim).item()
        n_nets = 5
        n_quantiles = 25
        top_quantiles_to_drop = 2 * n_nets
        

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
        self.actor.eval()

    def action_scaleup(self, action):
        low = self.min_action
        high = self.max_action
        a = np.zeros_like(low) -1.0
        b = np.zeros_like(low) +1.0
        action = low + (high - low)*((action - a)/(b - a))
        action = np.clip(action, low, high)
        return action

    # def action_scaledown(self, action):
    #     low = self.min_action
    #     high = self.max_action
    #     a = np.zeros_like(low) -1.0
    #     b = np.zeros_like(low) +1.0
    #     action = a + (b - a)*((action - low)/(high - low))
    #     action = np.clip(action, a, b)
    #     return torch.from_numpy(action).to(device)

    def draw_action(self, state):
        self.actor.eval()
        norm_state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.action_scaleup(self.actor(norm_state)[0][0].cpu().detach().numpy())
        des_pos = np.array([action[0],action[1],0.1645])                                #'ee_desired_height': 0.1645
        x_ = [action[0],action[1]] 
        y = self.get_ee_pose(state)[0][:2]
        des_v = action[2]*(x_-y)/(np.linalg.norm(x_-y)+1e-8)
        des_v = np.concatenate((des_v,[0])) 
        _,x = inverse_kinematics(self.robot_model, self.robot_data,des_pos)
        # _,x = solve_hit_config_ik_null(self.robot_model,self.robot_data, des_pos, des_v, self.get_joint_pos(state))
        return x

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
        self.critic.load_state_dict(torch.load(filename + "_critic",map_location=device))
        self.critic_target.load_state_dict(torch.load(filename + "_critic_target",map_location=device))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer",map_location=device))
        self.actor.load_state_dict(torch.load(filename + "_actor",map_location=device))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer",map_location=device))
        self.log_alpha = torch.load(filename + '_log_alpha',map_location=device)
        self.alpha_optimizer.load_state_dict(torch.load(filename + "_alpha_optimizer",map_location=device))

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
