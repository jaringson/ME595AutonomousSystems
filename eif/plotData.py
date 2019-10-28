import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from IPython.core.debugger import set_trace

plt.ion()  # enable interactive drawing


class plotData:
    '''
        This class plots the time histories for the pendulum data.
    '''

    def __init__(self):
        # Number of subplots = num_of_rows*num_of_cols
        self.num_rows = 3    # Number of subplot rows
        self.num_cols = 1    # Number of subplot columns

        # Crete figure and axes handles
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True)
        self.fig2, self.ax2 = plt.subplots(self.num_rows, self.num_cols, sharex=True)

        # Instantiate lists to hold the time and data histories
        self.time_history = []  # time
        self.x_history = []
        self.y_history = []
        self.theta_history = []
        self.x_hat_history = []
        self.y_hat_history = []
        self.theta_hat_history = []

        # create a handle for every subplot.
        self.handle = []
        self.handle.append(myPlot(self.ax[0], ylabel='x (m)', title='Midterm States', legend=['Truth', 'Estimate']))
        self.handle.append(myPlot(self.ax[1], ylabel='y (m)'))
        self.handle.append(myPlot(self.ax[2], xlabel='t (s)', ylabel='theta (deg)'))


        self.error_x = []
        self.error_y = []
        self.error_theta = []
        self.sig_x = []
        self.sig_y = []
        self.sig_theta = []

        self.handle2 = []
        self.handle2.append(myPlot(self.ax2[0], ylabel='X Error', title='Midterm Errors', legend=['Error', '2 Sig', '-2 Sig']))
        self.handle2.append(myPlot(self.ax2[1], ylabel='Y Error'))
        self.handle2.append(myPlot(self.ax2[2], xlabel='t(s)', ylabel='Theta Error'))

    def update(self, t, states, mu, sig):
        '''
            Add to the time and data histories, and update the plots.
        '''
        # update the time history of all plot variables
        self.time_history.append(t)  # time
        self.x_history.append(states[0])
        self.y_history.append(states[1])
        self.theta_history.append(180.0/np.pi*states[2])
        self.x_hat_history.append(mu[0])
        self.y_hat_history.append(mu[1])
        self.theta_hat_history.append(180.0/np.pi*mu[2])

        # update the plots with associated histories
        # self.handle[0].update(self.time_history, [self.theta_history, self.theta_ref_history])
        self.handle[0].update(self.time_history, [self.x_history, self.x_hat_history])
        self.handle[1].update(self.time_history, [self.y_history, self.y_hat_history])
        self.handle[2].update(self.time_history, [self.theta_history, self.theta_hat_history])

        self.error_x.append(states[0]-mu[0])
        self.error_y.append(states[1]-mu[1])
        self.error_theta.append(states[2]-mu[2])
        self.sig_x.append(2*np.sqrt(sig[0][0]))
        self.sig_y.append(2*np.sqrt(sig[1][1]))
        self.sig_theta.append(2*np.sqrt(sig[2][2]))

        self.handle2[0].update(self.time_history, [self.error_x, self.sig_x, [-x for x in self.sig_x]])
        self.handle2[1].update(self.time_history, [self.error_y, self.sig_y, [-y for y in self.sig_y]])
        self.handle2[2].update(self.time_history, [self.error_theta, self.sig_theta, [-th for th in self.sig_theta]])


class myPlot:
    '''
        Create each individual subplot.
    '''
    def __init__(self, ax,
                 xlabel='',
                 ylabel='',
                 title='',
                 legend=None):
        '''
            ax - This is a handle to the  axes of the figure
            xlable - Label of the x-axis
            ylable - Label of the y-axis
            title - Plot title
            legend - A tuple of strings that identify the data.
                     EX: ("data1","data2", ... , "dataN")
        '''
        self.legend = legend
        self.ax = ax                  # Axes handle
        self.colors = ['b', 'g', 'g', 'c', 'm', 'y', 'b']
        # A list of colors. The first color in the list corresponds
        # to the first line object, etc.
        # 'b' - blue, 'g' - green, 'r' - red, 'c' - cyan, 'm' - magenta
        # 'y' - yellow, 'k' - black
        self.line_styles = ['-', '-', '-', '-.', ':']
        # A list of line styles.  The first line style in the list
        # corresponds to the first line object.
        # '-' solid, '--' dashed, '-.' dash_dot, ':' dotted

        self.line = []

        # Configure the axes
        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)

        # Keeps track of initialization
        self.init = True

    def update(self, time, data):
        '''
            Adds data to the plot.
            time is a list,
            data is a list of lists, each list corresponding to a line on the plot
        '''
        if self.init == True:  # Initialize the plot the first time routine is called
            for i in range(len(data)):
                # Instantiate line object and add it to the axes
                self.line.append(Line2D(time,
                                        data[i],
                                        color=self.colors[np.mod(i, len(self.colors) - 1)],
                                        ls=self.line_styles[np.mod(i, len(self.line_styles) - 1)],
                                        label=self.legend[i] if self.legend != None else None))
                self.ax.add_line(self.line[i])
            self.init = False
            # # add legend if one is specified
            # if self.legend != None:
            #     plt.legend(handles=self.line)
        else: # Add new data to the plot
            # Updates the x and y data of each line.
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        # Adjusts the axis to fit all of the data
        self.ax.relim()
        self.ax.autoscale()
        self.ax.legend()
