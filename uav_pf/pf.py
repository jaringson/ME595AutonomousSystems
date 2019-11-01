import numpy as np
# import numpy.linalg
# import numpy.linalg.cholesky as chol
from IPython.core.debugger import set_trace
from importlib import reload

from copy import deepcopy
import vaporlite_parameters as MAV
from FixedWing import FixedWing

from tools import normalize
from tools import boxminus



class PF:
    def __init__(self):

        self.mu = np.array([[MAV.pn0],  # (0)
                               [MAV.pe0],   # (1)
                               [MAV.pd0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0], # (12)
                               [1]])   # (13)
        # self.mu = np.array([[0],[0],[0]])

        self.fw = FixedWing()

        self.M = 100

        state_max = np.array([[5],  # (0)
                               [5],   # (1)
                               [5],   # (2)

                               [6],    # (3)
                               [1],    # (4)
                               [1],    # (5)

                               [1],    # (6)
                               [1],    # (7)
                               [1],    # (8)
                               [1],    # (9)

                               [np.pi/1],    # (10)
                               [np.pi/1],    # (11)
                               [np.pi/1],    # (12)

                               [1.0/self.M]])         # (13)
        state_min = -state_max
        state_min[13] = 1.0/self.M

        self.Chi = (state_max-state_min) * np.random.random((14,self.M)) + state_min

    def run(self, state, u, dt):
        Chi_bar = np.array([])
        # for m in range(self.M):
        # for m in range(self.M):
        # set_trace()
        temp_u = np.atleast_2d(u).T + np.random.normal(0, 1, (4,self.M))
        # self.Chi[0:13:] = normalize(self.Chi[0:13,:])
        x = self.fw.forward_simulate_dt(self.Chi[0:13,:], temp_u, dt)

        w = self.measurement_prob(state, x)

        # if (any(w==0)):
        #     # print(w)
        #     w[w==0] += 1e-100
        #     # print(w)
        #     # set_trace()

        Chi_bar = np.vstack((x,w))

        # print(w)
        # print(Chi_bar[13,:])
        Chi_bar[13,:] = Chi_bar[13,:]/np.sum(Chi_bar[13,:])
        # print(Chi_bar[13,:])
        # set_trace()
        self.Chi = self.low_variance_sampler(Chi_bar)
        self.mu = np.atleast_2d(np.average(self.Chi[0:13,:],axis=1)).T



    def low_variance_sampler(self, Chi):
        Chi_bar = np.array([])
        r = np.random.random()*1.0/self.M
        c = Chi[13,0]
        i = 0
        for m in range(self.M):
            U = r + m * 1.0/self.M
            while U > c:
                i = i+1
                c += Chi[13,i]
            if Chi_bar.size == 0:
                Chi_bar = Chi[:,i]
            else:
                Chi_bar = np.vstack((Chi_bar,Chi[:,i]))

        Chi_bar = Chi_bar.T
        Chi_bar[3,:] = np.ones(self.M) + 1.0/self.M
        return Chi_bar

    def prob(self,a,b):
        return 1.0/(np.sqrt(2*np.pi*b*b))*np.exp(-0.5*a*a/(b*b))

    def measurement_prob(self, true_state, state):
        pos = self.prob(true_state[0]-state[0], 0.5)*self.prob(true_state[1]-state[1], 0.5)*self.prob(true_state[2]-state[2], 0.5)
        # print(pos)
        # set_trace()
        # dm = boxminus(np.atleast_2d(true_state[6:10]).T,state[6:10])
        rot = 1.0 #self.prob(dm[0],0.5)*self.prob(dm[1],0.5)*self.prob(dm[2],0.5)
        return pos*rot

    def get_mu(self):
        return self.mu

    def get_std(self):
        return np.std(self.Chi,axis=1)

    def get_particles(self):
        return self.Chi
