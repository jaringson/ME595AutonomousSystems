import numpy as np
import matplotlib.pyplot as plt

from IPython.core.debugger import set_trace
from importlib import reload

from pf import PF
from FixedWing import FixedWing

data = np.load("gather4.npz")
pf = PF()
fw = FixedWing()

truth = np.array([])
estimate = np.array([])
time_array = []
time = 0

dt = 0

fig = plt.figure(10)
plt.ion()
ax = fig.add_subplot(111, projection='3d')
plt.pause(0.01)

for i in range(data['x'].shape[1]):
    dt += data['dt'][i]
    time += data['dt'][i]
    if dt > 0.1:
        pf.run(data['x'][:,i], data['u'][:,i], dt)

        dt = 0

        time_array.append(time)
        if truth.size == 0:
            truth = data['x'][:,i]
            estimate = pf.get_mu()
        else:
            truth = np.vstack((truth,data['x'][:,i]))
            estimate = np.hstack((estimate,pf.get_mu()))
        # set_trace()
        fw.visualize(pf.get_mu(),pf.get_particles(),ax)
        # fw.visualize(np.atleast_2d(data['x'][:,i]).T,ax)

    print(time)
    if time > 2:
        break
truth = truth.T

fig, axes = plt.subplots(3, 1, sharey=True)
for i in range(3):
    # set_trace()
    axes[i].plot(time_array,truth[i,:]-estimate[i,:])
plt.show()
plt.pause(100)
