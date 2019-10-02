import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from IPython.core.debugger import set_trace
from importlib import reload

import params
reload(params)
import params as P

class Animation:
    '''
        Create pendulum animation
    '''
    def __init__(self):
        self.flagInit = True
        # self.fig, self.ax = plt.subplots()
        self.handle = []

        self.fig = plt.figure(1)
        self.ax = self.fig.add_subplot(111)
        # self.ax2 = self.fig.add_subplot(122)
        # self.fig.subplots_adjust(bottom=0.25)

        self.set_up_ax(self.ax)
        self.plotObjects = []


    def set_up_ax(self, ax, title='2D Test'):
        maxAxis = 10

        ax.set_xlim([-maxAxis, maxAxis])
        # ax.set_xlim([-20, -2])
        ax.set_xlabel('X')

        ax.set_ylim([-maxAxis, maxAxis])
        # ax.set_ylim([-8, 8])
        ax.set_ylabel('Y')

        ax.set_title(title)


    def draw(self, state):

        self.drawVehicle(state)


        if self.flagInit:
            self.flagInit = False


    def drawVehicle(self, state, color="limegreen"):
        x_pos = state[0]
        y_pos = state[1]
        theta = state[2]
        radius = 0.5

        xy = (x_pos,y_pos)

        X = [x_pos, x_pos+np.cos(theta)]
        Y = [y_pos, y_pos+np.sin(theta)]

        if self.flagInit == True:
            self.handle.append(mpatches.CirclePolygon(xy,
                radius = radius, resolution = 15,
                fc = color, ec = 'black'))
            self.ax.add_patch(self.handle[0])

            line, = self.ax.plot(X,Y,lw = 1, c = 'red')
            self.handle.append(line)
        else:
            self.handle[0]._xy=xy

            self.handle[1].set_xdata(X)
            self.handle[1].set_ydata(Y)
