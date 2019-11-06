import numpy as np
from IPython.core.debugger import set_trace
from importlib import reload

import params
reload(params)
import params as P

class EKF_SLAM:
    def __init__(self):
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

        self.landmarks = [lm1, lm2, lm3, lm4, lm5, lm6, lm7,
            lm8, lm9, lm10, lm11, lm12, lm13, lm14,
            lm15, lm16, lm17, lm18, lm19]

        # self.Sig = np.diag((1,1,0.1))

        self.N = len(self.landmarks)
        # self.mu = np.zeros((3+2*self.N,1))
        self.mu = np.vstack((np.array([[P.x0],[P.y0],[P.theta0]]), np.zeros((2*self.N, 1))))
        self.Sig = np.block([[np.diag((0,0,0)), np.zeros((3,2*self.N))],
                [np.zeros((2*self.N,3)), np.eye(2*self.N)*1e5]])


        self.F = np.block([np.eye(3), np.zeros((3,len(self.landmarks)*2))])
        # set_trace()

    def run(self, state, u):
        self.propagate(u)
        for i  in range(len(self.landmarks)):
            self.update(state, i)

    def propagate(self, u):
        v = u[0]
        omega = u[1]
        theta = self.mu[2][0]
        vo = v/omega
        G = np.eye(3+2*self.N) + self.F.T @ np.array([
            [0, 0, -vo * np.cos(theta) + vo * np.cos(theta+omega*P.Ts)],
            [0, 0, -vo * np.sin(theta) + vo * np.sin(theta+omega*P.Ts)],
            [0, 0, 0]
            ]) @ self.F

        V = np.array([
            [(-np.sin(theta)+np.sin(theta+omega*P.Ts))/omega, (v*(np.sin(theta)-np.sin(theta+omega*P.Ts)))/omega**2 + (v*np.cos(theta+omega*P.Ts)*P.Ts)/omega],
            [(np.cos(theta)-np.cos(theta+omega*P.Ts))/omega, -(v*(np.cos(theta)-np.cos(theta+omega*P.Ts)))/omega**2 + (v*np.sin(theta+omega*P.Ts)*P.Ts)/omega],
            [0, P.Ts]
            ])

        M = np.array([
            [P.alpha1*v**2+P.alpha2*omega**2, 0],
            [0, P.alpha3*v**2+P.alpha4*omega**2]
            ])

        R = V @ M @ V.T

        self.mu = self.mu + self.F.T @ np.array([
            [-vo * np.sin(theta) + vo * np.sin(theta+omega*P.Ts)],
            [vo * np.cos(theta) - vo * np.cos(theta+omega*P.Ts)],
            [omega*P.Ts]])

        self.Sig = G @ self.Sig @ G.T + self.F.T @ R @ self.F
        # set_trace()


    def update(self, state, index):

        landmark = self.landmarks[index]
        beta = np.radians(45)


        range = np.sqrt((landmark[0]-state[0])**2+(landmark[1]-state[1])**2) + np.random.normal(0,P.sig_r)
        bearing = np.arctan2(landmark[1]-state[1],landmark[0]-state[0])-state[2] + np.random.normal(0,P.sig_phi)
        z = np.array([
            [range],
            [bearing]
            ])

        # print(bearing)
        if(np.abs(self.wrap_angle(bearing)) < beta/2.0):
            # print(index)
            land_mu = self.mu[3+2*index:3+2*index+2]
            if(land_mu == 0).all():
                temp = np.array([[range * np.cos(bearing + self.mu[2][0])],
                                                [range * np.sin(bearing + self.mu[2][0])]])
                land_mu = self.mu[0:2] + temp
                self.mu[3+2*index:3+2*index+2] = land_mu

            delta = np.array([[land_mu[0][0] - self.mu[0][0]],
                                [land_mu[1][0] - self.mu[1][0]]])

            q = delta.T @ delta
            q = q[0][0]
            # z += np.random.multivariate_normal(np.array([0.0,0.0]), self.Q, 1).T

            # q = (landmark[0]-self.mu[0][0])**2+(landmark[1]-self.mu[1][0])**2
            z_hat = np.array([
                [np.sqrt(q)],
                [np.arctan2(delta[1][0],delta[0][0])-self.mu[2][0]]
                ])


            Fx = np.block([ [np.eye(3), np.zeros((3,2*self.N))],
                            [np.zeros((2,3)), np.zeros((2,2*(index+1)-2)), np.eye(2), np.zeros((2,2*self.N-2*(index+1)))]
                ])


            H = 1.0/q * np.array([[-np.sqrt(q)*delta[0][0],  -np.sqrt(q)*delta[1][0], 0, np.sqrt(q)*delta[0][0], np.sqrt(q)*delta[1][0]],
                [delta[1][0], -delta[0][0], -q, -delta[1][0], delta[0][0]]
                ])

            H = H @ Fx

            K = self.Sig @ H.T @ np.linalg.inv(H @ self.Sig @ H.T + self.Q)

            res = z - z_hat
            res[1][0] = self.wrap_angle(res[1][0])

            self.mu = self.mu + K @ (res)
            self.Sig = (np.eye(2*self.N+3) - K @ H) @ self.Sig
            # print(self.Sig)
            # set_trace()

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
