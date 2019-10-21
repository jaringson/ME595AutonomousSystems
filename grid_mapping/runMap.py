import numpy as np
import scipy
import scipy.io as sio
from IPython.core.debugger import set_trace

class GridCell:
    def __init__(self, x, y):
        self.prob = 0.5
        self.x = x
        self.y = y

data = sio.loadmat('state_meas_data.mat')


def inverse_range_sensor(m,x,y,theta,z):

    alpha = 1
    beta = 5
    z_max = 150

    r = np.sqrt((m.x-x)**2+(m.y-y)**2)
    phi = np.arctan2(m.y-y,m.x-x)-theta
    # k = np.argmin(np.abs(phi-data['thk']))
    k = np.argmin(np.abs(phi-z[1]))

    l0 = np.log(0.5/(1-0.5))
    if r > np.min([z_max, z[:,k][0]+alpha/2.0]) or np.abs(phi-z[:,k][1]) > beta/2.0:
        return l0
    if z[:,k][0] < z_max and np.abs(r-z[:,k][0]) < alpha/2.0:
        return np.log(0.7/(1-0.7))
    if r <= z[:,k][0]:
        return np.log(0.3/(1-0.3))

grid = []
for i in range(100):
    for j in range(100):
        cell = GridCell(i*0.5+0.5,j*0.5+0.5)
        grid.append(cell)

for i in range(data['z'].shape[2]):
    for cell in grid:
        x,y,theta = data['X'][:,i]
        # set_trace()
        cell.prob = cell.prob + inverse_range_sensor(cell,x,y,theta,data['z'][:,:,i])
