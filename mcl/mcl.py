import numpy as np
# import numpy.linalg
# import numpy.linalg.cholesky as chol
from IPython.core.debugger import set_trace
from importlib import reload

from copy import deepcopy

import params
reload(params)
import params as P

class Particle:
    def __init__(self,x,w):
        self.x = x
        self.w = w

class MCL:
    def __init__(self):
        self.Q = np.eye(2)
        self.Q[0][0] = P.sig_r**2
        self.Q[1][1] = P.sig_phi**2

        lm1 = [6,4]
        lm2 = [-7,8]
        lm3 = [6,-4]

        self.landmarks = [lm1, lm2, lm3]

        self.K = np.zeros((3,2))
        self.Sig = np.diag((0.01,0.01,0.01))
        self.mu = np.array([[-5],[-3],[np.pi/2]])
        # self.mu = np.array([[0],[0],[0]])



        self.M = 1000

        a = np.array([[-5],[-5],[-np.pi],[1.0/self.M]]) + np.vstack((self.mu,0))
        b = np.array([[5],[5],[np.pi],[1.0/self.M]]) + np.vstack((self.mu,0))
        self.Chi = (b-a) * np.random.random((4,self.M)) + a

    def run(self, state, u):
        Chi_bar = np.array([])
        for m in range(self.M):

            v = u[0] + np.random.normal(0, np.sqrt(P.alpha1*u[0]**2+P.alpha2*u[1]**2))
            omega = u[1] + np.random.normal(0, np.sqrt(P.alpha3*u[0]**2+P.alpha4*u[1]**2))

            x = self.propagateDynamics([v,omega],self.Chi[0:3,m])
            w = 0
            for i in range(3):
                w += self.measurement_prob(state, x, self.landmarks[i])
            # w += self.measurement_prob(state,x,self.landmarks[0])
            if Chi_bar.size == 0:
                Chi_bar = np.hstack((x,w))
            else:
                Chi_bar = np.vstack((Chi_bar,np.hstack((x,w))))

        Chi_bar = Chi_bar.T
        Chi_bar[3,:] = Chi_bar[3,:]/np.sum(Chi_bar[3,:])

        # set_trace()
        self.Chi = self.low_variance_sampler(Chi_bar)
        self.mu = np.atleast_2d(np.average(self.Chi[0:3,:],axis=1)).T


    def low_variance_sampler(self, Chi):
        Chi_bar = np.array([])
        r = np.random.random()*1.0/self.M
        c = Chi[3,0]
        i = 0
        for m in range(self.M):
            U = r + m * 1.0/self.M
            while U > c:
                i = i+1
                c += Chi[3,i]
            if Chi_bar.size == 0:
                Chi_bar = Chi[:,i]
            else:
                Chi_bar = np.vstack((Chi_bar,Chi[:,i]))
            # set_trace()

        return Chi_bar.T


    def propagateDynamics(self, u, state):
        v = u[0]
        omega = u[1]
        theta = state[2]
        vo = v/omega
        r1 = -vo * np.sin(theta) + vo * np.sin(theta+omega*P.Ts)
        # print(r1)
        r2 = vo * np.cos(theta) - vo * np.cos(theta+omega*P.Ts)
        r3 = omega*P.Ts
        # set_trace()
        state = deepcopy(state) + np.array([r1,r2,r3]) #+ np.random.normal([0,0,0], [0.1,0.1,0.1])
        return state

    def prob(self,a,b):
        return 1.0/(np.sqrt(2*np.pi*b*b))*np.exp(-0.5*a*a/(b*b))

    def measurement_prob(self, true_state, state, landmark):
        r = np.sqrt((landmark[0]-state[0])**2+(landmark[1]-state[1])**2)
        phi = np.arctan2(landmark[1]-state[1],landmark[0]-state[0])

        r_hat = np.sqrt((landmark[0]-true_state[0])**2+(landmark[1]-true_state[1])**2) + np.random.normal(0,P.sig_r)
        phi_hat = np.arctan2(landmark[1]-true_state[1],landmark[0]-true_state[0]) + np.random.normal(0,P.sig_phi)

        return self.prob(r-r_hat,P.sig_r) * self.prob(phi-phi_hat,P.sig_phi)

    def get_mu(self):
        return self.mu

    def get_std(self):
        return np.std(self.Chi,axis=1)

    def get_particles(self):
        return self.Chi
