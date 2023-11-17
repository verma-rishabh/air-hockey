import numpy as np
import torch
import argparse
import os
import numpy as np
from casadi import SX, vertcat, Function, sumsqr ,nlpsol
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian

import utils
# import OurDDPG
# import DDPG
from air_hockey_challenge.framework.air_hockey_challenge_wrapper import AirHockeyChallengeWrapper
from air_hockey_agent.agent_builder import build_agent


def custom_rewards(base_env, state, action, next_state, absorbing):
    print(base_env, state, action, next_state, absorbing)
    print(base_env.info)

env = AirHockeyChallengeWrapper(env="7dof-hit", interpolation_order=3, debug=True)
policy = build_agent(env.env_info)



P = SX.sym('P',3)
V = SX.sym('V',3)

des_state = [1,1,0.1645]

X = vertcat(P, V)

theta = SX.sym('theta',7)
theta_dot = SX.sym('theta_dot',7)

u = vertcat(theta)

# p,r = forward_kinematics(policy.robot_model, policy.robot_data,des_pos)
F_forward = Function("F_forward", [theta],[forward_kinematics(policy.robot_model, policy.robot_data,u)[0]])

f = sumsqr(P-des_state)
g = P[2]-0.1645
# Define number of steps in the control horizon and discretization step
N = 2
delta_t = 1/50
# Define RK4 integrator function and initial state x0_bar
x0_bar = [0, 0]

# Define the weighting matrix for the cost function
# Q = np.eye(6)
# R = np.eye(n_a)


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
Xk = SX.sym('X0', )
w += [Xk]
lbw += x0_bar    # set initial state
ubw += x0_bar    # set initial state
w0 += x0_bar     # set initial state

# Formulate the NLP
for k in range(N):
    # New NLP variable for the control
    Uk = SX.sym('U_' + str(k))
    w   += [Uk]
    lbw += [-1]
    ubw += [1]
    w0  += [0]

    # Integrate till the end of the interval
    #Xk_end = F_rk4(Xk, Uk)
    #J = J + delta_t * 1/2 * (Xk.T @ Q @ Xk + R * Uk**2) # Complete with the stage cost

    # New NLP variable for state at end of interval
    Xk = SX.sym(f'X_{k+1}', 2)
    w += [Xk]
    lbw += [-np.pi/2, -inf]
    ubw += [2*np.pi, inf]
    w0 += [0, 0]

    # Add equality constraint to "close the gap" for multiple shooting
    #g   += [Xk_end-Xk]
    lbg += [0, 0]
    ubg += [0, 0]
J = J + 1/2 * (Xk.T @ Q @ Xk) # Complete with the terminal cost (NOTE it should be weighted by delta_t)

# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
solver = nlpsol('solver', 'ipopt', prob)

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()