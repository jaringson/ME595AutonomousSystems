import numpy as np
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace
from importlib import reload

import sys
sys.path.append("../ekf_slam")

import params as P
from dynamics import Dynamics
from animation import Animation
from plotData import plotData
from fast_slam import FAST_SLAM

from copy import deepcopy

dynamics = Dynamics()
dataPlot = plotData()
fast_slam = FAST_SLAM()
animation = Animation(fast_slam)

estimate = np.array([])
actual = np.array([])

t = P.t_start
while t < P.t_end:
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        t = t + P.Ts
        vc = 1.25+0.5*np.cos(2*np.pi*(0.2)*t)
        omegac = -0.5+0.2*np.cos(2*np.pi*(0.6)*t)
        noise_v = vc + np.random.normal(0, np.sqrt(P.alpha1*vc**2+P.alpha2*omegac**2))
        noise_omega = omegac + np.random.normal(0, np.sqrt(P.alpha3*vc**2+P.alpha4*omegac**2))

        u = [noise_v,noise_omega]
        dynamics.propagateDynamics(u)
        fast_slam.run(dynamics.states(), u)
        animation.draw(dynamics.states(), fast_slam)

        if estimate.size == 0:
            estimate = np.array(fast_slam.get_mu())
            actual = np.array(dynamics.states())
        else:
            estimate = np.vstack((estimate, fast_slam.get_mu()))
            actual = np.vstack((actual, dynamics.states()))

    dataPlot.update(t, dynamics.states(), fast_slam.get_mu(), fast_slam.get_sig())


    plt.pause(0.001)

# fig = plt.figure(6)
# Sig = deepcopy(ekf_slam.Sig)
# plt.imshow(Sig)

# Keeps the program from closing until the user presses a button.
# print('Press key to close')
# plt.waitforbuttonpress()
# plt.close()

# fig = plt.figure(2)
# plt.title('2D')
# # plt.plot(data['X_tr'][0], data['X_tr'][1], label='Truth')
# plt.plot(actual[:,0], actual[:,1], label='Truth')
# plt.plot(estimate[:,0], estimate[:,1], label='Estimate')
# # plt.scatter(data['m'][0], data['m'][1], marker='*', label='Landmarks')
# # plt.legend(loc=(0.5,0.5))
# plt.xlabel('x (m)')
# plt.ylabel('y (m)')
# plt.show()

plt.pause(100)
