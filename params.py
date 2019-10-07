import numpy as np

# Simulation Parameters
t_start = 0.0  # Start time of simulation
t_end = 10.0  # End time of simulation
Ts = 0.1  # sample time for simulation
t_plot = 0.1  # the plotting and animation is updated at this rate


# vehicleSpeed = 5.0
x0 = -5
y0 = -3
theta0 = np.pi/2

mass = 5.0
b = 2.0

kp = 50
ki = 0.1

fMax = 200.0
fMin = -200.0

alpha1 = 0.1
alpha2 = 0.01
alpha3 = 0.01
alpha4 = 0.1

sig_r = 0.1
sig_phi = 0.05
