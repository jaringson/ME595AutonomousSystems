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

        self.K = np.zeros((3,2))
        self.Sig = np.diag((0.01,0.01,0.01))
        self.mu = np.array([[0],[0],[0]])

        self.n = 3
        self.L = self.n*2+1

        alpha = 0.4
        kappa = 4
        beta = 2
        self.lam = alpha**2*(self.L+kappa)-self.L
        self.gamma = np.sqrt(self.L+self.lam)

        self.wm = [self.lam/(self.L+self.lam)]
        self.wc = [self.lam/(self.L+self.lam) + (1-alpha**2+beta)]
        for i in range(1,2*self.L+1):
            self.wm.append(1.0/(2*(self.L+self.lam)))
            self.wc.append(1.0/(2*(self.L+self.lam)))

        # self.wm = np.array([self.wm])
        # self.wc = np.array([self.wc])

        self.Chi_x = []
        self.Chi_u = []
        self.Chi_z = []

        self.Sig_bar = []
        self.mu_bar = []


    def run(self, state, u):
        self.propagate(u)
        for i in range(3):
            self.update(state.T.tolist()[0], self.landmarks[i])
            self.get_sigma_points(u)
            self.use_sigma_points()


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

        self.get_sigma_points(u)

        input_u = np.array([u]).T + self.Chi_u
        # set_trace()
        Chi_x_bar = np.zeros_like(self.Chi_x)
        for i in range(self.Chi_u.shape[1]):
            Chi_x_bar[:,i] = self.propagateDynamics(input_u[:,i], self.Chi_x[:,i])

        self.Chi_x = Chi_x_bar

        self.use_sigma_points()


    def get_sigma_points(self, u):
        v = u[0]
        omega = u[1]
        M = np.array([
            [P.alpha1*v**2+P.alpha2*omega**2, 0],
            [0, P.alpha3*v**2+P.alpha4*omega**2]
            ])

        mu_a = np.vstack((self.mu, np.zeros((4,1))))

        Sig_a = np.block([  [self.Sig,np.zeros((3,4))],
            [np.zeros((2,3)),M,np.zeros((2,2))],
            [np.zeros((2,5)),self.Q]])

        # print(Sig_a)

        Chi_a = np.block([ mu_a, mu_a+self.gamma*np.linalg.cholesky(Sig_a), mu_a-self.gamma*np.linalg.cholesky(Sig_a)])
        self.Chi_x = Chi_a[:3,:]
        self.Chi_u = Chi_a[3:5,:]
        self.Chi_z = Chi_a[5:,:]

    def use_sigma_points(self):


        self.mu_bar = np.atleast_2d(np.sum(np.multiply(self.wm,self.Chi_x),axis=1)).T
        self.Sig_bar = np.zeros((3,3))
        for i in range(2*self.L+1):
            self.Sig_bar += self.wc[i]*(self.Chi_x[:,i]-self.mu_bar.T).T@(self.Chi_x[:,i]-self.mu_bar.T)

    def measurement(self, state, landmark):
        z = np.array([
            np.sqrt((landmark[0]-state[0])**2+(landmark[1]-state[1])**2),
            np.arctan2(landmark[1]-state[1],landmark[0]-state[0])-state[2]
            ])
        return z


    def update(self, state, landmark):
        z = np.array([
            [np.sqrt((landmark[0]-state[0])**2+(landmark[1]-state[1])**2) + np.random.normal(0,P.sig_r)],
            [np.arctan2(landmark[1]-state[1],landmark[0]-state[0])-state[2] + np.random.normal(0,P.sig_phi)]
            ])
        # z += np.random.multivariate_normal(np.array([0.0,0.0]), self.Q, 1).T

        Z_bar = np.zeros((2,self.Chi_x.shape[1]))
        for i in range(self.Chi_x.shape[1]):
            Z_bar[:,i] = self.measurement(self.Chi_x[:,i],landmark) + self.Chi_z[:,i]

        z_hat = np.atleast_2d(np.sum(np.multiply(self.wm,Z_bar),axis=1)).T

        S = np.zeros((2,2))
        Sig_x_z = np.zeros((3,2))
        for i in range(2*self.L+1):
            S += self.wc[i]*(Z_bar[:,i]-z_hat.T).T @ (Z_bar[:,i]-z_hat.T)
            Sig_x_z += self.wc[i]*(self.Chi_x[:,i]-self.mu_bar.T).T @ (Z_bar[:,i]-z_hat.T)

        self.K = Sig_x_z @ np.linalg.inv(S)

        # print(self.mu)
        # set_trace()
        self.mu = self.mu_bar + self.K @ (z-z_hat)
        self.Sig = self.Sig_bar - self.K @ S @ self.K.T

    def get_mu(self):
        return self.mu

    def get_sig(self):
        return np.array([[self.Sig[0,0]],[self.Sig[1,1]], [self.Sig[2,2]]])

    def get_k(self):
        return self.K.reshape((6,1))
