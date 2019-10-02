import numpy as np
import pdb
import matplotlib.pyplot as plt


for t in time:
    u = 0
    if t >= 0 and t < 5:
        u = 50
    elif t >= 25 and t < 30:
        u =-50
    # x = propagateDynamics(x,u,dt)
    x = sysd.A@x+sysd.B*u
