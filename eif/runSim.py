import numpy as np
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace
from importlib import reload
import scipy
import scipy.io as sio

import params as P
from plotData import plotData
from eif import EIF

# animation = Animation()
dataPlot = plotData()
eif = EIF()


estimate = np.array([])

data = sio.loadmat('data.mat')

for i in range(data['t'].shape[1]):
    time = data['t'][:,i][0]
    v = data['v_c'][:,i][0]
    om = data['om_c'][:,i][0]
    range = data['range_tr'][i]
    bearing = data['bearing_tr'][i]

    eif.run(v, om, range, bearing)


    dataPlot.update(time, data['X_tr'][:,i], eif.get_mu(), eif.get_sig())

    if estimate.size == 0:
        estimate = np.array(eif.get_mu())
    else:
        estimate = np.vstack((estimate, eif.get_mu()))

    # plt.pause(0.001)



fig = plt.figure(3)
plt.title('Midterm 2D Plot')
plt.plot(data['X_tr'][0], data['X_tr'][1], label='truth')
plt.plot(estimate[:,0], estimate[:,1], label='estimate')
plt.scatter(data['m'][0], data['m'][1], marker='*')
plt.legend()
plt.show()

plt.pause(1000)
