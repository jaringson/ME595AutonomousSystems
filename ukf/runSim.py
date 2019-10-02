import numpy as np
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace
from importlib import reload

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

import ukf
reload(ukf)
from ukf import UKF

dynamics = Dynamics()
animation = Animation()
# dataPlot = plotData()
ukf = UKF()

states = dynamics.states()
mu = ukf.get_mu()
sig = ukf.get_sig()
Ks = ukf.get_k()

t = P.t_start
while t < P.t_end:
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        t = t + P.Ts
        vc = 1+0.5*np.cos(2*np.pi*(0.2)*t)
        omegac = -0.2+2*np.cos(2*np.pi*(0.6)*t)
        noise_v = vc + np.random.normal(0, np.sqrt(P.alpha1*vc**2+P.alpha2*omegac**2))
        noise_omega = omegac + np.random.normal(0, np.sqrt(P.alpha3*vc**2+P.alpha4*omegac**2))

        u = [noise_v,noise_omega]
        dynamics.propagateDynamics(u)
        animation.draw(dynamics.states())
        ukf.run(dynamics.states(), u)

        states = np.hstack((states, dynamics.states()))
        mu = np.hstack((mu, ukf.get_mu()))
        sig = np.hstack((sig, ukf.get_sig()))
        Ks = np.hstack((Ks, ukf.get_k()))

    # dataPlot.update(t, dynamics.states(), ukf.get_mu(), ukf.get_sig())

    plt.pause(0.0001)
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
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), 2*np.sqrt(sig[0,:]), label='2 Sigma', color='green')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), -2*np.sqrt(sig[0,:]), color='green')
ax.legend()

ax = fig.add_subplot(312)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), states[1,:] - mu[1,:], label='Error')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), 2*np.sqrt(sig[1,:]), label='2 Sigma', color='green')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), -2*np.sqrt(sig[1,:]), color='green')

ax = fig.add_subplot(313)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), states[2,:] - mu[2,:], label='Error')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), 2*np.sqrt(sig[2,:]), label='2 Sigma', color='green')
ax.plot(np.linspace(0,P.t_end,num=states.shape[1])  , -2*np.sqrt(sig[2,:]), color='green')

fig = plt.figure(4)

ax = fig.add_subplot(611)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), Ks[0,:])
ax = fig.add_subplot(612)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), Ks[1,:])
ax = fig.add_subplot(613)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), Ks[2,:])
ax = fig.add_subplot(614)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), Ks[3,:])
ax = fig.add_subplot(615)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), Ks[4,:])
ax = fig.add_subplot(616)
ax.plot(np.linspace(0,P.t_end,num=states.shape[1]), Ks[5,:])

plt.show()
