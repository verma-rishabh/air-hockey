import numpy as np
import torch
from air_hockey_challenge.utils.kinematics import forward_kinematics, inverse_kinematics, jacobian, link_to_xml_name
import mujoco
import time

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e7)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size,1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
    
    def save(self,filename):
        np.savez(filename,state = self.state[:self.size,::],action = self.action[:self.size,::],\
            next_state = self.next_state[:self.size,::],reward = self.reward[:self.size,::],not_done =self.not_done[:self.size,::])
    
    def load(self,filename):
        x = np.load(filename)
        size = x["state"].shape[0]
        self.state[:size,:] = x["state"]
        self.next_state[:size,:] = x["next_state"]
        self.reward[:size,:] = x["reward"]
        self.not_done[:size,:] = x["not_done"]
        self.action[:size,:] = x["action"]
        self.ptr = size
        self.size = size

def solve_hit_config_ik_null(robot_model,robot_data, x_des, v_des, q_0, max_time=5e-3):
        t_start = time.time()
        reg = 0e-6
        dim = q_0.shape[0]
        IT_MAX = 1000
        eps = 1e-4
        damp = 1e-3
        progress_thresh = 20.0
        max_update_norm = 0.1
        i = 0
        TIME_MAX = max_time
        success = False

        dtype = robot_data.qpos.dtype

        robot_data.qpos = q_0

        # q_l = robot_model.jnt_range[:, 0]
        # q_h = robot_model.jnt_range[:, 1]
        q_l = np.array([-2.96706, -2.0944 , -2.96706, -2.0944 , -2.96706, -2.0944 , -3.05433])
        q_h = np.array([2.96706, 2.0944 , 2.96706, 2.0944 , 2.96706, 2.0944 , 3.05433])
        lower_limit = (q_l + q_h) / 2 - 0.95 * (q_h - q_l) / 2
        upper_limit = (q_l + q_h) / 2 + 0.95 * (q_h - q_l) / 2

        name = link_to_xml_name(robot_model, 'ee')

        def objective(q, grad):
            if grad.size > 0:
                grad[...] = numerical_grad(objective, q)
            f = v_des @ jacobian(robot_model, robot_data, q)[:3, :dim]
            return f @ f + reg * np.linalg.norm(q - q_0)

        null_opt_stop_criterion = False
        while True:
            # forward kinematics
            mujoco.mj_fwdPosition(robot_model, robot_data)

            x_pos = robot_data.body(name).xpos

            err_pos = x_des - x_pos
            error_norm = np.linalg.norm(err_pos)

            f_grad = numerical_grad(objective, robot_data.qpos.copy())
            f_grad_norm = np.linalg.norm(f_grad)
            if f_grad_norm > max_update_norm:
                f_grad = f_grad / f_grad_norm

            if error_norm < eps:
                success = True
            if time.time() - t_start > TIME_MAX or i >= IT_MAX or null_opt_stop_criterion:
                break

            jac_pos = np.empty((3, robot_model.nv), dtype=dtype)
            mujoco.mj_jacBody(robot_model, robot_data, jac_pos, None, robot_model.body(name).id)

            update_joints = jac_pos.T @ np.linalg.inv(jac_pos @ jac_pos.T + damp * np.eye(3)) @ err_pos

            # Add Null space Projection
            null_dq = (np.eye(robot_model.nv) - np.linalg.pinv(jac_pos) @ jac_pos) @ f_grad
            null_opt_stop_criterion = np.linalg.norm(null_dq) < 1e-4
            update_joints += null_dq

            update_norm = np.linalg.norm(update_joints)

            # Check whether we are still making enough progress, and halt if not.
            progress_criterion = error_norm / update_norm
            if progress_criterion > progress_thresh:
                success = False
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            mujoco.mj_integratePos(robot_model, robot_data.qpos, update_joints, 1)
            robot_data.qpos = np.clip(robot_data.qpos, lower_limit, upper_limit)
            i += 1
        q_cur = robot_data.qpos.copy()

        return success, q_cur
    
def numerical_grad(fun, q):
    eps = np.sqrt(np.finfo(np.float64).eps)
    grad = np.zeros_like(q)
    for i in range(q.shape[0]):
        q_pos = q.copy()
        q_neg = q.copy()
        q_pos[i] += eps
        q_neg[i] -= eps
        grad[i] = (fun(q_pos, np.array([])) - fun(q_neg, np.array([]))) / 2 / eps
    return grad