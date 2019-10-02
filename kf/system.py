import numpy as np
import pdb
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal

b = 20.0
m = 100.0
dt = 0.05

A = np.array([[0,1],[0,-b/m]])
B = np.array([[0],[1.0/m]])
C = np.array([[1,0]])
D = np.array([0])

sys = sp.signal.StateSpace(A, B, C, D)
sysd = sys.to_discrete(dt)

def get_system():
    return sysd.A,sys.B,sysd.C,sysd.D,dt
