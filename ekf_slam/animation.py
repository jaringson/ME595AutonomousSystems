import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from IPython.core.debugger import set_trace
from importlib import reload

import params as P

from estimator_type import Estimator

class Animation:
    '''
        Create pendulum animation
    '''
    def __init__(self, estimator):
        self.flagInit = True
        # self.fig, self.ax = plt.subplots()
        self.handle = []

        self.fig = plt.figure(3)
        self.ax = self.fig.add_subplot(111)
        # self.ax2 = self.fig.add_subplot(122)
        # self.fig.subplots_adjust(bottom=0.25)

        self.index = 0

        self.set_up_ax(self.ax, estimator)
        self.plotObjects = []


    def set_up_ax(self, ax, estimator):
        maxAxis = 20

        ax.set_xlim([-maxAxis, maxAxis])
        # ax.set_xlim([-20, -2])
        ax.set_xlabel('X')

        ax.set_ylim([-maxAxis, maxAxis])
        # ax.set_ylim([-8, 8])
        ax.set_ylabel('Y')

        ax.set_title(estimator.title)


    def draw(self, state, estimator):
        landmarks = estimator.landmarks
        mu = estimator.get_mu()

        self.drawVehicle(state)
        self.drawLandmarks(landmarks)

        if estimator.est_type == Estimator.EKF_SLAM:
            sig = estimator.Sig
            self.drawEKFSLAMLandmarkEstimates(mu, sig)

        if estimator.est_type == Estimator.FAST_SLAM:
            Chi = estimator.Chi
            landmarks_mu = estimator.landmarks_mu
            # self.drawVehicle(Chi[:,0], color="red", r=0.1)
            # self.drawLandmarks(landmarks_mu[0].reshape((estimator.N,2)), color="blue")
            for i in range(Chi.shape[1]):
                self.drawVehicle(Chi[:,i], color="red", r=0.1)
                self.drawLandmarks(landmarks_mu[i].reshape((estimator.N,2)), color="blue", r=0.05)
                # self.index += 1

        self.index = 0


        if self.flagInit:
            self.flagInit = False


    def drawVehicle(self, state, color="limegreen", r=0.5):
        x_pos = state[0]
        y_pos = state[1]
        theta = state[2]
        radius = r

        xy = (x_pos,y_pos)

        X = [x_pos, x_pos+np.cos(theta)]
        Y = [y_pos, y_pos+np.sin(theta)]

        if self.flagInit == True:
            self.handle.append(mpatches.CirclePolygon(xy,
                radius = radius, resolution = 15,
                fc = color, ec = 'black'))
            self.ax.add_patch(self.handle[self.index])

            line, = self.ax.plot(X,Y,lw = 1, c = 'red')
            self.handle.append(line)
        else:
            self.handle[self.index]._xy=xy

            self.handle[self.index+1].set_xdata(X)
            self.handle[self.index+1].set_ydata(Y)
        self.index += 2


    def drawLandmarks(self, landmarks, color="red", r=0.2):
        N = len(landmarks)
        # set_trace()
        for i in range(N):
            # index = 2 + i
            xy = ( landmarks[i][0], landmarks[i][1] )

            radius = r

            if self.flagInit == True:
                self.handle.append(mpatches.CirclePolygon(xy,
                    radius = radius, resolution = 15,
                    fc = color, ec = 'black'))
                self.ax.add_patch(self.handle[self.index])
            else:
                self.handle[self.index]._xy=xy
            self.index += 1


    def drawEKFSLAMLandmarkEstimates(self, mu, sig, color="blue"):
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
            a = chisquare_val*np.sqrt(w[k])
            b = chisquare_val*np.sqrt(w[1-k])
            # print(a,b,w)
            # print(land_sig)
            # set_trace()
            # index = 2 + N + 2*i
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
                self.ax.add_patch(self.handle[self.index])

                self.handle.append(mpatches.Ellipse(xy,
                    width = 2*a, height = 2*b, angle = angle,
                    fc = 'gray', ec = 'black', alpha=0.1))

                self.ax.add_patch(self.handle[self.index+1])
                # self.ax.add_patch(self.handle[self.index+1])
            else:
                self.handle[self.index]._xy=xy

                self.handle[self.index+1].set_center(xy)
                self.handle[self.index+1].width=2*a
                self.handle[self.index+1].height=2*b
                self.handle[self.index+1].angle=angle
            self.index += 2
