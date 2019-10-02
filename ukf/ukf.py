import numpy as np
# import numpy.linalg
# import numpy.linalg.cholesky as chol
from IPython.core.debugger import set_trace
from importlib import reload

import params
reload(params)
import params as P

class UKF:
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

        self.n = 3
        L = 7

        alpha = 0.4
        kappa = 4
        beta = 2
        self.gamma = alpha**2*(self.n+kappa)-self.n

        self.wm = [self.gamma/(L+self.gamma)]
        self.wc = [self.gamma/(L+self.gamma) + (1-alpha**2+beta)]
        for i in range(1,2*L+1):
            self.wm.append(1/(2*(L+self.gamma)))
            self.wc.append(1/(2*(L+self.gamma)))

        # self.wm = np.array([self.wm])
        # self.wc = np.array([self.wc])


    def run(self, state, u):
        self.propagate(u)
        for i  in range(3):
            self.update(state, self.landmarks[i])


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
        state = state + np.array([r1,r2,r3])
        return state

    def propagate(self, u):
        v = u[0]
        omega = u[1]
        theta = self.mu[2][0]
        vo = v/omega


        M = np.array([
            [P.alpha1*v**2+P.alpha2*omega**2, 0],
            [0, P.alpha3*v**2+P.alpha4*omega**2]
            ])

        mu_a = np.vstack((self.mu, np.zeros((4,1))))

        Sig_a = np.block([  [self.Sig,np.zeros((3,4))], [np.zeros((2,3)),M,np.zeros((2,2))],  [np.zeros((2,5)),self.Q]])

        Chi_a = np.block([ mu_a, mu_a+self.gamma*np.linalg.cholesky(Sig_a), mu_a-self.gamma*np.linalg.cholesky(Sig_a)])
        Chi_x = Chi_a[:3,:]
        Chi_u = Chi_a[3:5,:]
        Chi_z = Chi_a[5:,:]

        input_u = np.array([u]).T + Chi_u
        Chi_x_next = np.zeros_like(Chi_x)
        for i in range(Chi_u.shape[1]):
            Chi_x_next[:,i] = self.propagateDynamics(input_u[:,i], Chi_x[:,i])
        set_trace()

        self.mu = np.atleast_2d(np.sum(np.multiply(self.wm,Chi_x_next),axis=1)).T
        intermediate = ((Chi_x_next-self.mu).T@(Chi_x_next-self.mu))
        self.Sig =
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
