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

import plotData
reload(plotData)
from plotData import plotData

import ekf
reload(ekf)
from ekf import EKF

dynamics = Dynamics()
animation = Animation()
dataPlot = plotData()
ekf = EKF()

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
        ekf.run(dynamics.states(), u)

    dataPlot.update(t, dynamics.states(), ekf.get_mu(), ekf.get_sig())

    plt.pause(0.001)

# Keeps the program from closing until the user presses a button.
# print('Press key to close')
# plt.waitforbuttonpress()
# plt.close()

plt.pause(100)
