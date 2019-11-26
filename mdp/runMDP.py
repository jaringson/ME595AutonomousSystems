import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from IPython.core.debugger import set_trace

from copy import deepcopy

N = 100
Np = 100 + 2

map = np.zeros((Np,Np))

# Initialize walls and obstacle maps as empty
walls = np.zeros((Np,Np))
obs1 = np.zeros((Np,Np))
obs2 = np.zeros((Np,Np))
obs3 = np.zeros((Np,Np))
goal = np.zeros((Np,Np))

# Create exterior walls
walls[1,1:N] = -100
walls[1:N,1] = -100
walls[N,1:N] = -100
walls[1:N+1,N] = -100

# Create single obstacle
obs1[19:41,29:79] = -5000
obs1[9:19,59:64] = -5000

# Another obstacle
obs2[44:64,9:44] = -5000

# Another obstacle
obs3[42:99,74:84] = -5000
obs3[69:79,49:74] = -5000

# The goal states
goal[74:79,95:97] = 100000

# Put walls and obstacles into map
orig_map = walls + obs1 + obs2 + obs3 + goal
map[orig_map == 0] = -2

sub_map = map[1:N+1,1:N+1]
mask = sub_map == -2

# fig = plt.figure(1)
# image = plt.imshow(np.flipud(map.T))
# fig.colorbar(image)
# plt.show()

error = 10000
map_d1 = deepcopy(map)

plot = 10000
counter = 0

while error > 1e-4:
    north = map[2:N+2,1:N+1]
    east = map[1:N+1,2:N+2]
    south = map[0:N,1:N+1]
    west = map[1:N+1,0:N]

    m_north = -2 + 0.8 * north + 0.1 * east + 0.1 * west
    m_east = -2 + 0.8 * east + 0.1 * north + 0.1 * south
    m_south = -2 + 0.8 * south + 0.1 * east + 0.1 * west
    m_west = -2 + 0.8 * west + 0.1 * north + 0.1 * south

    n_e = np.maximum(m_north, m_east)
    s_w = np.maximum(m_south, m_west)
    max = np.maximum(n_e, s_w)

    map[1:N+1,1:N+1] = orig_map[1:N+1,1:N+1] + max*mask
    # print(np.sum(map), np.sum(map_d1))
    error = np.abs(np.sum(map-map_d1))
    # print(error)
    map_d1 = deepcopy(map)

    if counter > plot:
        fig = plt.figure(1)
        image = plt.imshow(np.flipud(map.T))
        fig.colorbar(image)
        plt.show()
        counter = 0

    counter += 1

north = map[2:N+2,1:N+1]
east = map[1:N+1,2:N+2]
south = map[0:N,1:N+1]
west = map[1:N+1,0:N]

m_north = 0.8 * north + 0.1 * east + 0.1 * west
m_east = 0.8 * east + 0.1 * north + 0.1 * south
m_south = 0.8 * south + 0.1 * east + 0.1 * west
m_west = 0.8 * west + 0.1 * north + 0.1 * south

indices = np.argmax(np.dstack((m_north,m_south,m_east,m_west)),2)
indices[mask == False] = -1

fig = plt.figure(1)
image = plt.imshow(map.T, origin='lower')
fig.colorbar(image)
i = 0
for row in indices:
    j = 0
    for value in row:
        if value == 0:
            plt.arrow(i+1, j+1, 0.25, 0, width=0.1)
        if value == 1:
            plt.arrow(i+1, j+1, -0.25, 0, width=0.1)
        if value == 2:
            plt.arrow(i+1, j+1, 0, 0.25, width=0.1)
        if value == 3:
            plt.arrow(i+1, j+1, 0, -0.25, width=0.1)
        j += 1
    i += 1

current = np.array([28,20])
all_x = []
all_y = []
while True:
    i = current[0]
    j = current[1]
    if indices[i,j] == -1:
        break
    if indices[i,j] == 0:
        current[0] += 1
    if indices[i,j] == 1:
        current[0] -= 1
    if indices[i,j] == 2:
        current[1] += 1
    if indices[i,j] == 3:
        current[1] += 1

    all_x.append(current[0]+1)
    all_y.append(current[1]+1)
plt.plot(all_x, all_y)

plt.show()
