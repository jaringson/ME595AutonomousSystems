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
        maxAxis = 20

        ax.set_xlim([-maxAxis, maxAxis])
        # ax.set_xlim([-20, -2])
        ax.set_xlabel('X')

        ax.set_ylim([-maxAxis, maxAxis])
        # ax.set_ylim([-8, 8])
        ax.set_ylabel('Y')

        ax.set_title(title)


    def draw(self, state, ekf_slam):
        landmarks = ekf_slam.landmarks
        mu = ekf_slam.get_mu()
        sig = ekf_slam.Sig

        self.drawVehicle(state)
        self.drawLandmarks(landmarks)
        self.drawLandmarkEstimates(mu, sig)


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


    def drawLandmarks(self, landmarks, color="red"):
        N = len(landmarks)
        for i in range(N):
            index = 2 + i
            xy = ( landmarks[i][0], landmarks[i][1] )

            radius = 0.2

            if self.flagInit == True:
                self.handle.append(mpatches.CirclePolygon(xy,
                    radius = radius, resolution = 15,
                    fc = color, ec = 'black'))
                self.ax.add_patch(self.handle[index])
            else:
                self.handle[index]._xy=xy


    def drawLandmarkEstimates(self, mu, sig, color="blue"):
        N = ( len(mu)-3 ) // 2

        for i in range(N):
            land_sig = sig[3+i:3+i+2,3+i:3+i+2]
            [w,v] = np.linalg.eig(land_sig)
            k = np.argmax(w)
            large_v = v[:,k]
            angle = np.arctan2(large_v[1],large_v[0])
            if angle < 0:
                angle += 2*np.pi
            angle = np.degrees(angle)
            chisquare_val = 2.4477 # 95% Confidence Interval
            a = chisquare_val*np.sqrt(w[0])
            b = chisquare_val*np.sqrt(w[1])
            # print(a,b,w)
            # print(land_sig)
            # set_trace()
            index = 2 + N + 2*i
            radius = 0.2

            # if a > 10:
            #     a = 0
            #     b = 0
            # if b > 10:
            #     b = 0
            #     a = 0

            xy = ( mu[3+2*i], mu[3+2*i+1] )

            if xy[0] == 0 and xy[1] == 0:
                a = 0
                b = 0
                xy = (100,100)

            if self.flagInit == True:
                self.handle.append(mpatches.CirclePolygon(xy,
                    radius = radius, resolution = 15,
                    fc = color, ec = 'black'))
                self.ax.add_patch(self.handle[index])

                self.handle.append(mpatches.Ellipse(xy,
                    width = 2*a, height = 2*b, angle = angle,
                    fc = 'gray', ec = 'black', alpha=0.1))

                self.ax.add_patch(self.handle[index])
                self.ax.add_patch(self.handle[index+1])
            else:
                self.handle[index]._xy=xy

                self.handle[index+1].set_center(xy)
                self.handle[index+1].width=2*a
                self.handle[index+1].height=2*b
                self.handle[index+1].angle=angle
