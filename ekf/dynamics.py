import numpy as np
from IPython.core.debugger import set_trace
from importlib import reload

import params
reload(params)
import params as P


class Dynamics:
    '''
        Model the physical system
    '''

    def __init__(self):
        # Initial state conditions
        self.state = np.array([[P.x0],
                                [P.y0],
                                [P.theta0]])



    def propagateDynamics(self, u):
        v = u[0]
        omega = u[1]
        theta = self.state[2][0]
        vo = v/omega
        # set_trace()
        r1 = -vo * np.sin(theta) + vo * np.sin(theta+omega*P.Ts)
        # print(r1)
        r2 = vo * np.cos(theta) - vo * np.cos(theta+omega*P.Ts)
        r3 = omega*P.Ts
        self.state = self.state + np.array([[r1],[r2],[r3]])

        # self.state[0] += np.random.normal(0, np.sqrt(P.alpha1*v**2+P.alpha2*omega**2))
        # self.state[1] += np.random.normal(0, np.sqrt(P.alpha3*v**2+P.alpha4*omega**2))
        # self.wrap_angle()



    def states(self):
        # noise_x = self.state[0] + np.random.normal(0, np.sqrt(P.alpha1*v**2+P.alpha2*omega**2))
        # noise_y = self.state[1] + np.random.normal(0, np.sqrt(P.alpha3*v**2+P.alpha4*omega**2))
        return self.state.T.tolist()[0]

    def wrap_angle(self):
        while(self.state[2][0] < -np.pi):
            self.state[2][0] += 2*np.pi
        while(self.state[2][0] > np.pi):
            self.state[2][0] -= 2*np.pi
    # def pos_states(self):
    #     return np.array([self.state.item(0), self.state.item(1)])
    #
    # def vel_states(self):
    #     return np.array([self.state.item(2), self.state.item(3)])
    #
    # def pos_vel_states(self):
    #     return self.pos_states(), self.vel_states()
