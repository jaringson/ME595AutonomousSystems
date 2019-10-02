import numpy as np
# import pdb
from IPython.core.debugger import set_trace
import matplotlib.pyplot as plt
from importlib import reload

import scipy as sp
import scipy.signal

import system
reload(system)
from system import get_system

b = 20.0
m = 100.0
dt = 0.05

A = np.array([[0,1],[0,-b/m]])
B = np.array([[0],[1.0/m]])
C = np.array([[1,0]])
D = np.array([0])

sys = sp.signal.StateSpace(A, B, C, D)
sysd = sys.to_discrete(dt)

A = sysd.A
B = sysd.B
C = sysd.C


R = np.diag((0.0001,0.01))
Q = 0.001

end = 50.0
time = np.arange(0,end+dt,dt)
# print(t)

Sigma = np.diag((0.1,0.1))

x = np.array([0.0,0.0])
x_hat_init = np.array([0.0,0.0])
all_x = np.squeeze(x)
mu = np.random.multivariate_normal(x_hat_init, Sigma, 1).T
all_x_hat = np.squeeze(mu)
all_K = []

x = np.atleast_2d(x)
x = x.T

all_sig = Sigma.flatten()


for t in time:
    u = 0
    if 0 <= t < 5:
        u = 50
    elif 25 <= t < 30:
        u =-50
    # set_trace()
    x = A @ x + B * u + np.random.multivariate_normal(np.array([0.0,0.0]), R, 1).T

    all_x = np.vstack((all_x,np.squeeze(x)))

    # set_trace()
    mu = x
    z = C @ mu + np.random.normal(0,Q**0.5)

    mu = A @ mu + B * u
    Sigma = A @ Sigma @ A.T + R

    K = Sigma @ C.T @ np.linalg.inv(C @ Sigma @ C.T + Q)
    # # print(x+x_hat)
    mu = mu + K @ (z - C @ mu)
    Sigma = (np.eye(2)-K @ C) @ Sigma


    all_x_hat = np.vstack((all_x_hat, np.squeeze(mu)))
    all_sig = np.vstack((all_sig, Sigma.flatten()))

    if t == 0:
        all_K = np.squeeze(K)
    else:
        all_K = np.vstack((all_K,np.squeeze(K)))
    # x = mu

fig = plt.figure(1)
fig.clf()
ax = fig.add_subplot(211)
plt.plot(np.append(time,end+dt),all_x[:,0],label="Pos True")
plt.plot(np.append(time,end+dt),all_x_hat[:,0],label="Pos Estimate")
ax.legend()

ax = fig.add_subplot(212)
plt.plot(np.append(time,end+dt),all_x[:,1],label="Vel True")
plt.plot(np.append(time,end+dt),all_x_hat[:,1],label="Vel Estimate")
ax.legend()
fig.show()

fig = plt.figure(2)
fig.clf()
ax = fig.add_subplot(111)
plt.plot(time,all_K[:,0],label="K[0]")
plt.plot(time,all_K[:,1],label="K[1]")
ax.legend()
fig.show()

# set_trace()
fig = plt.figure(3)
fig.clf()
ax = fig.add_subplot(211)
plt.plot(np.append(time,end+dt),all_x[:,0]-all_x_hat[:,0],label="Pos Estimate")
plt.plot(np.append(time,end+dt),2*np.sqrt(all_sig[:,0]),color="orange",label="2 Sig")
plt.plot(np.append(time,end+dt),-2*np.sqrt(all_sig[:,0]),color="orange")
ax.legend()

ax = fig.add_subplot(212)
plt.plot(np.append(time,end+dt),all_x[:,1]-all_x_hat[:,1],label="Pos Estimate")
plt.plot(np.append(time,end+dt),2*np.sqrt(all_sig[:,3]),color="orange",label="2 Sig")
plt.plot(np.append(time,end+dt),-2*np.sqrt(all_sig[:,3]),color="orange")
ax.legend()

fig.show()

# def derivatives(state, u):
#
#     x = state[0,0]
#     v = state[1,0]
#
#     return np.array([[v],[(u-b*v)/m]])
#
#
# def propagateDynamics(state, u, dt): #, u=np.array([0.0,0.0])):
#         '''
#             Integrate the differential equations defining dynamics
#             P.Ts is the time step between function calls.
#             u contains the system input(s).
#         '''
#         # Integrate ODE using Runge-Kutta RK4 algorithm
#         # state = vehicle.get_states()
#         # u = vehicle.force
#         k1 = derivatives(state, u)
#         k2 = derivatives(state + dt/2*k1, u)
#         k3 = derivatives(state + dt/2*k2, u)
#         k4 = derivatives(state + dt*k3, u)
#         # print(P.Ts/6,k1,k2,k3,k4)
#         state += dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)
#         return state


# def kalman(sysd, mu, sig, u, z):
#     mubar = sysd.A*u+sysd.B*u
#     sigbar = sysd.A*
