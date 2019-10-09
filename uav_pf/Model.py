import numpy as np
from copy import deepcopy
from scipy.linalg import expm

class Model():

    def __init__(self):
        self.numStates = None
        self.numInputs = None

    def forward_simulate_dt(self,x,u,dt):
        print("forward_simulate_dt function not yet implemented.")

    def visualize(x):
        print("visualize function not yet implemented.")

    def calc_A_B(self,x,u,dt,delta=1.0e-4):
        x = deepcopy(x)
        u = deepcopy(u)
        A = np.zeros([self.numStates,self.numStates])
        B = np.zeros([self.numStates,self.numInputs])

        xdot0 = self.forward_simulate_dt(x,u,dt)
        xup = deepcopy(x)
        uup = deepcopy(u)
        xdown = deepcopy(x)
        udown = deepcopy(u)

        for i in xrange(0,self.numStates):
            # if i < self.numStates/2.0:
            #     delta = delta
            # else:
            #     delta = delta*10.0
            xup[i] = x[i] + delta
            xdown[i] = x[i] - delta
            xdotup = self.forward_simulate_dt(xup,u,dt).flatten()
            xdotdown = self.forward_simulate_dt(xdown,u,dt).flatten()
            A[:,i] = (xdotup-xdotdown)/(2.0*delta)
            xup[i] = x[i]
            xdown[i] = x[i]

        # delta = delta*100.0
        for i in xrange(0,self.numInputs):
            uup[i] = u[i] + delta
            udown[i] = u[i] - delta
            xdotup = self.forward_simulate_dt(x,uup,dt).flatten()
            xdotdown = self.forward_simulate_dt(x,udown,dt).flatten()
            B[:,i] = (xdotup-xdotdown)/(2.0*delta)
            uup[i] = u[i]
            udown[i] = u[i]

        return A,B

    def discretize_A_and_B(self,A,B,dt):

        Ad = expm(A*dt)
        try:
            Bd = np.matmul(np.linalg.inv(A),np.matmul(Ad-np.eye(Ad.shape[0]),B))
        except:
            Bd = B*dt
        return Ad,Bd
