import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from numpy.linalg import norm
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits import mplot3d
import pdb
from stl import mesh

from IPython.core.debugger import set_trace


class Animation:
    '''
        Create pendulum animation
    '''
    def __init__(self):
        self.flagInit = True
        # self.fig, self.ax = plt.subplots()
        self.handle = []

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # self.ax2 = self.fig.add_subplot(122)
        # self.fig.subplots_adjust(bottom=0.25)

        self.set_up_ax(self.ax)
        self.plotObjects = []

    def set_up_ax(self, ax, title='Grid Mapping'):
        maxAxis = 100

        ax.set_xlim([0, maxAxis])
        # ax.set_xlim([-20, -2])
        ax.set_xlabel('X')

        ax.set_ylim([0, maxAxis])
        # ax.set_ylim([-8, 8])
        ax.set_ylabel('Y')

        ax.set_title(title)

    def drawAll(self, state, grid):
        self.drawGrid(grid, 0)
        self.drawVehicle(state, 10000)
        self.drawHeading(state, 10001)
        if self.flagInit:
            self.flagInit = False

    def drawVehicle(self, state, idx, color="limegreen"):
        x_pos, y_pos, theta = state
        radius = 2

        xy = (x_pos,y_pos)

        if self.flagInit == True:
            self.handle.append(mpatches.CirclePolygon(xy,
                radius = radius, resolution = 15,
                fc = color, ec = 'black'))
            self.ax.add_patch(self.handle[idx])
        else:
            self.handle[idx]._xy=xy

    def drawHeading(self, state, idx, color="black"):
        x_pos, y_pos, theta = state

        # set_trace()
        X = [x_pos,x_pos+(3)*np.cos(theta)]
        Y = [y_pos,y_pos+(3)*np.sin(theta)]

        if self.flagInit == True:

            line, = self.ax.plot(X,Y,lw = 1, c = color)
            self.handle.append(line)
        else:
            self.handle[idx].set_xdata(X)
            self.handle[idx].set_ydata(Y)

    def drawGrid(self, grid, idx):
        i = idx
        # set_trace()
        for cell in grid:
            # print(i)
            if self.flagInit == True:
                square = mpatches.Rectangle((cell.x-0.5,cell.y-0.5), 1, 1)
                square.set_color(str(1-np.exp(cell.prob)/(1+np.exp(cell.prob))))
                self.handle.append(square)
                self.ax.add_patch(square)
            else:
                # t = 1
                # self.handle[i].set_xdata(cell.prob)
                self.handle[i].set_color(str(1-np.exp(cell.prob)/(1+np.exp(cell.prob))))
            i = i + 1
