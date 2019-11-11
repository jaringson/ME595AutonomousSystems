import numpy as np
# import numpy.linalg
# import numpy.linalg.cholesky as chol
from IPython.core.debugger import set_trace
from importlib import reload

from copy import deepcopy

import sys
sys.path.append("../ekf_slam")

import params as P

from estimator_type import Estimator


class FAST_SLAM:
    def __init__(self):

        self.est_type = Estimator.FAST_SLAM

        self.Q = np.eye(2)
        self.Q[0][0] = P.sig_r**2
        self.Q[1][1] = P.sig_phi**2

        lm1 = [6,4]
        lm2 = [-7,8]
        lm3 = [12,-8]
        lm4 = [-2,0]
        lm5 = [-10,2]
        lm6 = [13,7]
        lm7 = [-7,-10]


        lm8 = [5,-5]
        lm9 = [-4,15]
        lm10 = [-3,-6]
        lm11 = [15,1]
        lm12 = [-1,12]
        lm13 = [13,17]
        lm14 = [-7,10]


        lm15 = [3,-10]
        lm16 = [1,-5]
        lm17 = [6,-15]
        lm18 = [2,10]
        lm19 = [-10,-10]

        # self.landmarks = [lm1]
        self.landmarks = [lm1, lm2, lm3, lm4, lm5, lm6, lm7,
            lm8, lm9, lm10, lm11, lm12, lm13, lm14,
            lm15, lm16, lm17, lm18, lm19]

        self.N = len(self.landmarks)

        self.mu = np.array([[P.x0],[P.y0],[P.theta0]])
        # self.mu = np.array([[0],[0],[0]])

        self.M = 100

        a = np.array([[-0],[-0],[0]]) + self.mu
        b = np.array([[0],[0],[0]]) + self.mu
        self.Chi = (b-a) * np.random.random((3,self.M)) + a

        self.landmarks_mu = np.full([self.M,2*self.N], np.nan)
        self.landmarks_sig = np.full([self.M,2*self.N,2*self.N], np.nan)

        self.weights = np.ones(self.M)* 1.0/self.M

        self.change = False

    def run(self, state, u):
        Chi_bar = np.array([])
        # for m in range(self.M):

            # v = u[0] + np.random.normal(0, np.sqrt(P.alpha1*u[0]**2+P.alpha2*u[1]**2))
            # omega = u[1] + np.random.normal(0, np.sqrt(P.alpha3*u[0]**2+P.alpha4*u[1]**2))

        self.Chi = self.propagateDynamics(u,self.Chi)

        for li in np.arange(self.N):
            landmark = self.landmarks[li]

            range = np.sqrt((landmark[0]-state[0])**2+(landmark[1]-state[1])**2) + np.random.normal(0,P.sig_r)
            bearing = np.arctan2(landmark[1]-state[1],landmark[0]-state[0])-state[2] + np.random.normal(0,P.sig_phi)
            z = np.array([
                [range],
                [bearing]
                ])

            beta = np.radians(45)

            if np.abs(self.wrap_angle(bearing)) < beta/2.0:


                for xi in np.arange(self.M):
                    x = self.Chi[:,xi]


                    # q = (mu[0]-x[0])**2+(mu[1]-x[1])**2)
                    # H = np.array([
                    #     [-(landmark[0]-x[0])/np.sqrt(q), -(landmark[1]-x[1])/np.sqrt(q)],
                    #     [(landmark[1]-x[1])/q, -(landmark[0]-x[0])/q]
                    #     ])
                    # Hinv = np.linalg.inv(H)
                    mu = []
                    sig = []

                    init = False

                    if np.isnan(self.landmarks_mu[xi,2*li:2*li+2]).any():
                        # t = 0
                        mu = x[0:2] + range * np.array([np.cos(bearing+x[2]),np.sin(bearing+x[2])])
                        init = True
                    else:
                        mu = self.landmarks_mu[xi,2*li:2*li+2]
                        sig = self.landmarks_sig[xi,2*li:2*li+2,2*li:2*li+2]


                    # set_trace()
                    q = (mu[0]-x[0])**2+(mu[1]-x[1])**2
                    H = np.array([
                        [-(mu[0]-x[0])/np.sqrt(q), -(mu[1]-x[1])/np.sqrt(q)],
                        [(mu[1]-x[1])/q, -(mu[0]-x[0])/q]
                        ])
                    Hinv = np.linalg.inv(H)

                    if init:

                        sig = Hinv @ self.Q @ Hinv.T
                        # sig = np.diag([1,1])

                        self.weights[xi] *= 1.0/self.M
                    else:

                        z_hat = np.array([
                            [np.sqrt(q)],
                            [np.arctan2(mu[1]-x[1],mu[0]-x[0])-x[2]]
                            ])

                        Q = H @ sig @ H.T + self.Q
                        K = sig @ H.T @ np.linalg.inv(Q)
                        res = -(z-z_hat)
                        res[1][0] = self.wrap_angle(res[1][0])
                        mu = mu + (K @ res).T

                        sig = (np.eye(2) - K @ H) @ sig

                        # print(res)
                        # print(sig)
                        # set_trace()
                        self.weights[xi] *= np.linalg.det(2*np.pi*Q)**-0.5 * np.exp(-0.5 * res.T @ np.linalg.inv(Q) @ res)

                    self.landmarks_mu[xi,2*li:2*li+2] = mu
                    self.landmarks_sig[xi,2*li:2*li+2,2*li:2*li+2] = sig

        self.weights /= np.sum(self.weights)
        # print(self.weights)
        # set_trace()
        self.low_variance_sampler()



        # # w = np.zeros(self.M)
        # w = np.ones(self.M)
        # for i in range(3):
        #     w_ret = self.measurement_prob(state, x, self.landmarks[i])
        #     # w_ret = w_ret/np.sum(w_ret)
        #     w *= w_ret
        # # w += self.measurement_prob(state,x,self.landmarks[0])
        # # set_trace()
        # Chi_bar = np.vstack((x,w))
        #
        # # Chi_bar = Chi_bar.T
        # # set_trace()
        # # if(np.isnan(np.sum(Chi_bar[3,:]/Chi_bar[3,:]))).any():
        # #     set_trace()
        # Chi_bar[3,:] = Chi_bar[3,:]/np.sum(Chi_bar[3,:])
        #
        # # set_trace()
        # # print(Chi_bar)
        # # if(self.change == True):
        # #     self.M = 100
        # self.Chi = self.low_variance_sampler(Chi_bar)
        # # set_trace()
        # # self.Chi[3,:] = np.ones(self.M)*1.0/self.M
        # self.mu = self.Chi[:,0]


    def low_variance_sampler(self):
        Chi_bar = np.array([])
        landmarks_mu = np.array([])
        landmarks_sig = np.array([])

        r = np.random.random()*1.0/self.M
        # set_trace()
        c = self.weights[0]
        i = 0
        for m in range(self.M):
            U = r + m * 1.0/self.M
            while U > c:
                i = i+1
                c += self.weights[i]
            if Chi_bar.size == 0:
                Chi_bar = self.Chi[:,i]
                landmarks_mu = self.landmarks_mu[i]
                landmarks_sig = self.landmarks_sig[i]
            else:
                Chi_bar = np.vstack((Chi_bar,self.Chi[:,i]))
                landmarks_mu = np.vstack((landmarks_mu, self.landmarks_mu[i]))
                landmarks_sig = np.dstack((landmarks_sig, self.landmarks_sig[i]))

        # Chi_bar = Chi_bar.T
        # set_trace()
        self.Chi = Chi_bar.T
        self.landmarks_mu = landmarks_mu
        self.landmarks_sig = landmarks_sig.T
        # Chi_bar[3,:] = np.ones(self.M) + 1.0/self.M
        self.weights = np.ones(self.M)
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

    # def wrap_angle(self, angles, amt=np.pi):
    #
    #     out = deepcopy(angles)
    #     while any(out < -amt):
    #         out = out + 2*np.pi * (out < -amt).astype(int)
    #     while any(out > amt):
    #         out = out - 2*np.pi * (out > amt).astype(int)
    #     return out

    def wrap_angle(self, angle):
        out = angle
        while(out < -np.pi):
            out += 2*np.pi
        while(out  > np.pi):
            out -= 2*np.pi
        return out

    def get_mu(self):
        return self.mu

    def get_std(self):
        return np.std(self.Chi,axis=1)

    def get_particles(self):
        return self.Chi
