import numpy as np
from IPython.core.debugger import set_trace
from importlib import reload

import params
reload(params)
import params as P

class EIF:
    def __init__(self):
        self.Q = np.eye(2)
        self.Q[0][0] = P.sig_r**2
        self.Q[1][1] = P.sig_phi**2

        self.M = np.array([
            [P.sig_v**2, 0],
            [0, P.sig_om**2]
            ])

        lm1 = [6,4]
        lm2 = [-7,8]
        lm3 = [12,-8]
        lm4 = [-2,0]
        lm5 = [-10,2]
        lm6 = [13,7]

        self.landmarks = [lm1, lm2, lm3, lm4, lm5, lm6]

        self.Sig = np.diag((1,1,0.1))
        self.mu = np.array([[-10],[-10],[0]])

        self.Om = np.linalg.inv(self.Sig)
        self.epsilon = self.Om @ self.mu

    def run(self, v, om, range, bearing):
        self.propagate(v, om)
        # set_trace()
        for i in np.arange(6):
            self.update(range[i], bearing[i], self.landmarks[i])

    def propagate(self, v, om):

        self.mu = self.Sig @ self.epsilon

        theta = self.mu[2][0]

        G = np.array([
            [1, 0, -v * np.sin(theta) * P.Ts],
            [0, 1,  v * np.cos(theta) * P.Ts],
            [0, 0, 1]
            ])

        V = np.array([
            [np.cos(theta), 0],
            [np.sin(theta), 0],
            [0, P.Ts]
            ])

        self.mu = self.mu + np.array([
            [v * np.cos(theta) * P.Ts],
            [v * np.sin(theta) * P.Ts],
            [om*P.Ts]])

        self.Sig = G @ self.Sig @ G.T + V @ self.M @ V.T
        self.Om = np.linalg.inv(self.Sig)
        self.epsilon = self.Om @ self.mu


    def update(self, range, bearing, truth):
        z = np.array([
            [range],
            [bearing]
            ])
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


        res = z - z_hat
        res[1][0] = self.wrap_angle(res[1][0])
        # set_trace()

        # S = H @ self.Sig @ H.T + self.Q
        # K = self.Sig @ H.T @ np.linalg.inv(S)
        #
        # self.mu = self.mu + K @ (res)
        # self.Sig = (np.eye(3) - K @ H) @ self.Sig

        self.Om = self.Om + H.T @ np.linalg.inv(self.Q) @ H
        self.epsilon = self.epsilon + H.T @ np.linalg.inv(self.Q) @ (res + H @ self.mu)

        self.Sig = np.linalg.inv(self.Om)
        self.mu = self.Sig @ self.epsilon

    def get_mu(self):
        return self.mu.T.tolist()[0]

    def get_sig(self):
        return self.Sig.tolist()

    def wrap_angle(self, angle):
        out = angle
        while(out < -np.pi):
            out += 2*np.pi
        while(out  > np.pi):
            out -= 2*np.pi
        return out
