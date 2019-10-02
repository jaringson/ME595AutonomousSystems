import numpy as np
from IPython.core.debugger import set_trace
from importlib import reload

import params
reload(params)
import params as P

class EKF:
    def __init__(self):
        self.Q = np.eye(2)
        self.Q[0][0] = P.sig_r**2
        self.Q[1][1] = P.sig_phi**2

        lm1 = [6,4]
        lm2 = [-7,8]
        lm3 = [6,-4]

        self.landmarks = [lm1, lm2, lm3]

        self.Sig = np.diag((1,1,0.1))

        self.mu = np.array([[-10],[-10],[0]])

    def run(self, state, u):
        self.propagate(u)
        for i  in range(3):
            self.update(state, self.landmarks[i])

    def propagate(self, u):
        v = u[0]
        omega = u[1]
        theta = self.mu[2][0]
        vo = v/omega
        G = np.array([
            [1, 0, -vo * np.cos(theta) + vo * np.cos(theta+omega*P.Ts)],
            [0, 1, -vo * np.sin(theta) + vo * np.sin(theta+omega*P.Ts)],
            [0, 0, 1]
            ])

        V = np.array([
            [(-np.sin(theta)+np.sin(theta+omega*P.Ts))/omega, (v*(np.sin(theta)-np.sin(theta+omega*P.Ts)))/omega**2 + (v*np.cos(theta+omega*P.Ts)*P.Ts)/omega],
            [(np.cos(theta)-np.cos(theta+omega*P.Ts))/omega, -(v*(np.cos(theta)-np.cos(theta+omega*P.Ts)))/omega**2 + (v*np.sin(theta+omega*P.Ts)*P.Ts)/omega],
            [0, P.Ts]
            ])

        M = np.array([
            [P.alpha1*v**2+P.alpha2*omega**2, 0],
            [0, P.alpha3*v**2+P.alpha4*omega**2]
            ])

        self.mu = self.mu + np.array([
            [-vo * np.sin(theta) + vo * np.sin(theta+omega*P.Ts)],
            [vo * np.cos(theta) - vo * np.cos(theta+omega*P.Ts)],
            [omega*P.Ts]])

        self.Sig = G @ self.Sig @ G.T + V @ M @ V.T


    def update(self, state, truth):
        z = np.array([
            [np.sqrt((truth[0]-state[0])**2+(truth[1]-state[1])**2) + np.random.normal(0,P.sig_r)],
            [np.arctan2(truth[1]-state[1],truth[0]-state[0])-state[2] + np.random.normal(0,P.sig_phi)]
            ])
        # set_trace()
        # z += np.random.multivariate_normal(np.array([0.0,0.0]), self.Q, 1).T

        q = (truth[0]-self.mu[0][0])**2+(truth[1]-self.mu[1][0])**2
        z_hat = np.array([
            [np.sqrt(q)],
            [np.arctan2(truth[1]-self.mu[1][0],truth[0]-self.mu[0][0])-self.mu[2][0]]
            ])

        H = np.array([
            [-(truth[0]-self.mu[0][0])/np.sqrt(q), -(truth[1]-self.mu[1][0])/np.sqrt(q), 0],
            [(truth[1]-self.mu[1][0])/q, -(truth[0]-self.mu[0][0])/q, -1]
            ])

        S = H @ self.Sig @ H.T + self.Q
        K = self.Sig @ H.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ (z-z_hat)
        self.Sig = (np.eye(3) - K @ H) @ self.Sig

    def get_mu(self):
        return self.mu.T.tolist()[0]

    def get_sig(self):
        return self.Sig.tolist()
