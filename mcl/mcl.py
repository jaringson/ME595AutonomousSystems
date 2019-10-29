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

        lm1 = [6,4]
        lm2 = [-7,8]
        lm3 = [6,-4]

        self.landmarks = [lm1, lm2, lm3]

        self.mu = np.array([[-5],[-3],[np.pi/2]])
        # self.mu = np.array([[0],[0],[0]])



        self.M = 1000

        a = np.array([[-5],[-5],[-np.pi],[1.0/self.M]]) + np.vstack((self.mu,0))
        b = np.array([[5],[5],[np.pi],[1.0/self.M]]) + np.vstack((self.mu,0))
        self.Chi = (b-a) * np.random.random((4,self.M)) + a


        self.change = False

    def run(self, state, u):
        Chi_bar = np.array([])
        # for m in range(self.M):

            # v = u[0] + np.random.normal(0, np.sqrt(P.alpha1*u[0]**2+P.alpha2*u[1]**2))
            # omega = u[1] + np.random.normal(0, np.sqrt(P.alpha3*u[0]**2+P.alpha4*u[1]**2))

        x = self.propagateDynamics(u,self.Chi[0:3,:])
        # w = np.zeros(self.M)
        w = np.ones(self.M)
        for i in range(3):
            w_ret = self.measurement_prob(state, x, self.landmarks[i])
            # w_ret = w_ret/np.sum(w_ret)
            w *= w_ret
        # w += self.measurement_prob(state,x,self.landmarks[0])
        # set_trace()
        Chi_bar = np.vstack((x,w))

        # Chi_bar = Chi_bar.T
        # set_trace()
        # if(np.isnan(np.sum(Chi_bar[3,:]/Chi_bar[3,:]))).any():
        #     set_trace()
        Chi_bar[3,:] = Chi_bar[3,:]/np.sum(Chi_bar[3,:])

        # set_trace()
        # print(Chi_bar)
        # if(self.change == True):
        #     self.M = 100
        self.Chi = self.low_variance_sampler(Chi_bar)
        # set_trace()
        # self.Chi[3,:] = np.ones(self.M)*1.0/self.M
        self.mu = np.atleast_2d(np.average(self.Chi[0:3,:],axis=1,weights=self.Chi[3,:])).T


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

        Chi_bar = Chi_bar.T
        Chi_bar[3,:] = np.ones(self.M) + 1.0/self.M
        return Chi_bar


    def propagateDynamics(self, u, state):
        # set_trace()
        v_hat = u[0] + np.random.normal(0, np.sqrt(P.alpha1*u[0]**2+P.alpha2*u[1]**2), (self.M))
        omega_hat = u[1] + np.random.normal(0, np.sqrt(P.alpha3*u[0]**2+P.alpha4*u[1]**2), (self.M))
        gamma_hat = np.random.normal(0, np.sqrt(P.alpha5*u[0]**2+P.alpha6*u[1]**2), (self.M))

        theta = state[2]
        vo = v_hat/omega_hat
        r1 = -vo * np.sin(theta) + vo * np.sin(theta+omega_hat*P.Ts)
        # print(r1)
        r2 = vo * np.cos(theta) - vo * np.cos(theta+omega_hat*P.Ts)
        r3 = omega_hat*P.Ts + gamma_hat*P.Ts
        # set_trace()
        state = deepcopy(state) + np.vstack((r1,r2,r3)) #+ np.random.normal([0,0,0], [0.1,0.1,0.1])
        return state

    def prob(self,a,b):
        return 1.0/(np.sqrt(2*np.pi*b*b))*np.exp(-0.5*a*a/(b*b))

    def measurement_prob(self, true_state, state, landmark):
        r = np.sqrt((landmark[0]-state[0])**2+(landmark[1]-state[1])**2)+ np.random.normal(0,P.sig_r)
        # phi = np.arctan2(landmark[1]-state[1],landmark[0]-state[0])
        phi = np.arctan2(landmark[1]-state[1],landmark[0]-state[0]) - state[2]+ np.random.normal(0,P.sig_phi)
        phi = self.wrap_angle(phi)

        r_hat = np.sqrt((landmark[0]-true_state[0])**2+(landmark[1]-true_state[1])**2) + np.random.normal(0,P.sig_r)
        # phi_hat = np.arctan2(landmark[1]-true_state[1],landmark[0]-true_state[0]) + np.random.normal(0,P.sig_phi)

        phi_hat = np.arctan2(landmark[1]-true_state[1],landmark[0]-true_state[0]) - true_state[2] + np.random.normal(0,P.sig_phi)
        # phi_hat = self.wrap_angle(phi_hat)

        # if np.abs(phi-phi_hat) > np.pi:
        # print(phi,phi_hat,np.pi)

        # set_trace()
        q = 0
        if(self.change):
            q = self.prob(r-r_hat,P.sig_r) * self.prob(self.wrap_angle(phi-phi_hat),P.sig_phi)
        else:
            q = self.prob(r-r_hat,np.sqrt(P.sig_r)) * self.prob(self.wrap_angle(phi-phi_hat),np.sqrt(P.sig_phi))
        # q = self.prob(r-r_hat,P.sig_r) * self.prob(phi-phi_hat,P.sig_phi)
        # print(np.max(weight))
        # print(r-r_hat)
        return q

    def wrap_angle(self, angles, amt=np.pi):

        out = deepcopy(angles)
        while any(out < -amt):
            out = out + 2*np.pi * (out < -amt).astype(int)
        while any(out > amt):
            out = out - 2*np.pi * (out > amt).astype(int)
        return out

    def get_mu(self):
        return self.mu

    def get_std(self):
        return np.std(self.Chi,axis=1)

    def get_particles(self):
        return self.Chi
