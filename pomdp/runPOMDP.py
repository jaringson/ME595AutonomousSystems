import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import permutations as prms

from IPython.core.debugger import set_trace

from copy import deepcopy

def max(eq, prob):
    res = eq @ prob
    plot = np.max(res)
    ind = np.argmax(res, 0)
    return np.unique(eq[ind], axis=0), plot

prob = np.linspace((0,1), (1,0), 1e4, axis=1)

msp1 = np.array([[0.7, 0],
                 [0, 0.3]])

msp2 = np.array([[0.3, 0],
                 [0, 0.7]])

p1p2 = np.array([[0.2, 0.8],
                 [0.8, 0.2]])

orig = np.array([[-100, 100],
               [100,  -50]])

eq = np.array([[-100, 100],
               [100,  -50],
               [-1,    -1]])
# print(prob)

eq, plot = max(eq, prob)

for i in np.arange(1,20):
    print("Iteration: ", i)
    res1 = eq @ msp1
    res2 = eq @ msp2
    perm =  list(prms(range(len(eq)),2))
    endpoints = [(ii,ii) for ii in range(len(eq))]
    ind = np.array(perm + endpoints)
    eq = res1[ind.T[0]] + res2[ind.T[1]]
    eq, plot = max(eq, prob)
    res4 = eq @ p1p2
    eq = np.array([*orig, *res4-1])
    eq, plot = max(eq, prob)

print(eq)
fig = plt.figure()
for i in range(len(eq)):
    # set_trace()
    plt.plot([0,1], eq[i])
    plt.plot(1, eq[i,1])
plt.show()
