import numpy as np
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace
from importlib import reload

import sys
sys.path.append("../")

import params
reload(params)
import params as P

import dynamics
reload(dynamics)
from dynamics import Dynamics

import animation
reload(animation)
from animation import Animation

# import plotData
# reload(plotData)
# from plotData import plotData

import mcl
reload(mcl)
from mcl import MCL

dynamics = Dynamics()
animation = Animation()
# dataPlot = plotData()
mcl = MCL()

states = dynamics.states()
mu = mcl.get_mu()
std = mcl.get_std()

t = P.t_start
while t < P.t_end:
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        t = t + P.Ts
        if t >= 1:
            mcl.change = True
        vc = 1+0.5*np.cos(2*np.pi*(0.2)*t)
        omegac = -0.2+2*np.cos(2*np.pi*(0.6)*t)
        noise_v = vc + np.random.normal(0, np.sqrt(P.alpha1*vc**2+P.alpha2*omegac**2))
        noise_omega = omegac + np.random.normal(0, np.sqrt(P.alpha3*vc**2+P.alpha4*omegac**2))

        u_noise = [noise_v,noise_omega]

        u_truth = [vc,omegac]
        dynamics.propagateDynamics(u_noise)
        animation.draw(dynamics.states(), Chi=mcl.get_particles())
        mcl.run(dynamics.states(), u_truth)

        states = np.hstack((states, dynamics.states()))
        mu = np.hstack((mu, mcl.get_mu()))
        std = np.vstack((std, mcl.get_std()))

    # dataPlot.update(t, dynamics.states(), ukf.get_mu(), ukf.get_sig())

    # plt.pause(0.0001)
# set_trace()

# Keeps the program from closing until the user presses a button.
# print('Press key to close')
# plt.waitforbuttonpress()
# plt.close()

fig = plt.figure(2)

ax = fig.add_subplot(311)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), states[0,:], label='True')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), mu[0,:], label='Estimated')
ax.legend()
# ax.set_xlabel('Max Velocity')
# ax.set_ylabel('Range')

ax = fig.add_subplot(312)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), states[1,:], label='True')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), mu[1,:], label='Estimated')

ax = fig.add_subplot(313)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), states[2,:], label='True')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), mu[2,:], label='Estimated')


fig = plt.figure(3)

ax = fig.add_subplot(311)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), states[0,:] - mu[0,:], label='Error')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), 2*std[:,0], label='2 Sigma', color='green')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), -2*std[:,0], color='green')
ax.set_ylim([-0.5,0.5])
ax.legend()

ax = fig.add_subplot(312)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), states[1,:] - mu[1,:], label='Error')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), 2*std[:,1], label='2 Sigma', color='green')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), -2*std[:,1], color='green')
ax.set_ylim([-0.5,0.5])

ax = fig.add_subplot(313)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), states[2,:] - mu[2,:], label='Error')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), 2*std[:,2], label='2 Sigma', color='green')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), -2*std[:,2], color='green')
ax.set_ylim([-0.5,0.5])



plt.show()
