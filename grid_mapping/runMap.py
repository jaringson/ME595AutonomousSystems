import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.io as sio
from IPython.core.debugger import set_trace

from plotter import Animation

import sys
# np.set_printoptions(threshold=sys.maxsize)

class GridCell:
    def __init__(self, x, y):
        self.prob = 0
        self.x = x
        self.y = y

data = sio.loadmat('state_meas_data.mat')

def draw(grid):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    cm = plt.cm.get_cmap('RdYlBu')
    # for cell in grid:
    # set_trace()
    # lp = cell.prob
    sc = plt.scatter([cell.x for cell in grid], [cell.y for cell in grid], c=[np.exp(cell.prob)/(1+np.exp(cell.prob)) for cell in grid],  s=35, cmap=cm)
    plt.colorbar(sc)
    # plt.show()
    plt.pause(0.001)


def inverse_range_sensor(grid,indices,xs,ys,x,y,theta,z,thk):

    # if np.isnan(z[0]).any():
    #     set_trace()
    z[1][np.isnan(z[0])] = thk.flatten()[np.isnan(z[0])]
    z[0][np.isnan(z[0])] = 1000000
    alpha = 2
    beta = 5
    z_max = 150

    r = np.sqrt((xs[indices]-x)**2+(ys[indices]-y)**2)
    phi = np.arctan2(ys[indices]-y,xs[indices]-x)-theta
    # k = np.argmin(np.abs(phi-data['thk']))
    z_range = np.repeat(z[0],phi.shape[0]).reshape((11,phi.shape[0]))
    z_bearing = np.repeat(z[1],phi.shape[0]).reshape((11,phi.shape[0]))

    bearing_pick = np.repeat(thk.flatten(),phi.shape[0]).reshape((11,phi.shape[0]))
    k = np.argmin(np.abs(phi-bearing_pick),axis=0)
    # if np.isnan(z).any():

    l0 = np.log(0.5/(1-0.5))


    z_r_sel  = z_range.T[np.arange(phi.shape[0]), k]
    z_b_sel  = z_bearing.T[np.arange(phi.shape[0]), k]
    # set_trace()

    # if r > np.minimum([np.ones_like(phi.shape[0])*z_max, r_sel+alpha/2.0]) or np.abs(phi-b_sel) > beta/2.0:
    #     return np.ones_like(grid[indices])*l0
    # temp = r > np.minimum(np.ones_like(phi.shape[0])*z_max, r_sel+alpha/2.0)
    f_ind = np.logical_or(r > np.minimum(np.ones_like(phi)*z_max, z_r_sel+alpha/2.0), np.abs(phi-z_b_sel) > beta/2.0)
    set_trace()
    temp = grid[indices]
    temp[f_ind] = l0
    grid[indices]= temp

    s_ind = np.logical_and(z_r_sel < np.ones_like(phi)*z_max, np.abs(r-z_r_sel) < alpha/2.0)
    temp = grid[indices]
    temp[s_ind] = np.log(0.8/(1-0.8))
    grid[indices] = temp
    # if temp[k][0] < z_max and np.abs(r-temo[k][0]) < alpha/2.0:
    #     return np.log(0.7/(1-0.7))

    # grid[indices][r <= r_sel] = np.log(0.4/(1-0.4))
    temp = grid[indices]
    temp[r <= z_r_sel] = np.log(0.4/(1-0.4))
    grid[indices] = temp

    print(grid[indices][f_ind])
    print(np.isnan(z[0]))

    # set_trace()
    return grid[indices]

    # if r <= temp[k][0]:
    #     return np.log(0.4/(1-0.4))
    # # set_trace()
    # return np.ones_like(grid[indices])*l0

num = 100

grid = np.zeros((num,num))
xs = np.repeat(np.linspace(0,99,num),num).reshape((num,num)).T
ys = np.repeat(np.linspace(0,99,num),num).reshape((num,num))

animation = Animation()

for i in range(data['z'].shape[2]):
    print(i)
    # if i % 50 == 0:
    #     animation.drawAll(data['X'][:,i], grid)
    #     plt.pause(0.01)
    #     # break
    x,y,theta = data['X'][:,i]
    phi = np.arctan2(ys-y,xs-x)-theta
    indices = np.logical_and(phi < np.pi/2.0, phi > -np.pi/2.0)
    grid[indices] = grid[indices] + inverse_range_sensor(grid, indices, xs, ys, x,y,theta ,data['z'][:,:,i], data['thk'])


# animation.drawAll(data['X'][:,-1], grid)
plt.imshow(np.exp(grid)/(1+np.exp(grid)), cmap='gray', vmin=0, vmax=1)
# plt.pause(0.01)
plt.show()
set_trace()
