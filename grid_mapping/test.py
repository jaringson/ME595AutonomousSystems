import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import scipy.io as sio
from IPython.core.debugger import set_trace

from plotter import Animation

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


def inverse_range_sensor(m,x,y,theta,z,thk):
    # if np.isnan(z).any():
    #     set_trace()

    z[1][np.isnan(z[0])] = thk.flatten()[np.isnan(z[0])]
    z[0][np.isnan(z[0])] = 100000000

    alpha = 1.0
    beta = np.radians(5)
    z_max = 150

    r = np.sqrt((m.x-x)**2+(m.y-y)**2)
    phi = np.arctan2(m.y-y,m.x-x)-theta
    # k = np.argmin(np.abs(phi-data['thk']))
    k = np.argmin(np.abs(phi-z[1]))

    l0 = np.log(0.5/(1-0.5))
    if np.isnan(z[0][k]):
        # set_trace()
        return l0

    if r > np.min([z_max, z[:,k][0]+alpha/2.0]) or np.abs(phi-thk.flatten()[k]) > beta/2.0:
        # set_trace()
        return l0
    if z[:,k][0] < z_max and np.abs(r-z[:,k][0]) < alpha/2.0:
        return np.log(0.7/(1-0.7))
    if r <= z[:,k][0]:
        return np.log(0.4/(1-0.4))
    # set_trace()
    # return l0

grid = []
for i in range(100):
    for j in range(100):
        cell = GridCell((j+1)-0.5,(i+1)-0.5)
        grid.append(cell)

animation = Animation()

for i in range(data['z'].shape[2]):
    print(i)
    # if i % 50 == 0:
    #     animation.drawAll(data['X'][:,i], grid)
    #     plt.pause(0.01)
        # break
    x,y,theta = data['X'][:,i]
    for cell in grid:
        phi = np.arctan2(cell.y-y,cell.x-x)-theta
        if phi < np.pi/2.0 and phi > -np.pi/2.0:
            # lp = np.log(cell.prob/(1-cell.prob)) + inverse_range_sensor(cell,x,y,theta,data['z'][:,:,i]) + 0.00001
            # set_trace()
            # cell.prob = np.exp(lp/(1+lp))
            cell.prob = cell.prob + inverse_range_sensor(cell,x,y,theta,data['z'][:,:,i],data['thk'])
animation.drawAll(data['X'][:,-1], grid)
plt.show()
# set_trace()
