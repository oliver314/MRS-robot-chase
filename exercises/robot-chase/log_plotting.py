from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt


if __name__ == '__main__':
    data = np.genfromtxt('/tmp/gazebo_exercise.txt', delimiter=',')

    startpoint = 10
    t = data[startpoint:, 0]
    error = []
    det = []
    sigma = []
    for i in range(3):
        error.append(data[startpoint:, 1+i*3])
        det.append(data[startpoint:, 2 + i * 3])
        sigma.append(data[startpoint:, 3 + i * 3])

    for b in range(3):
        fig, ax1 = plt.subplots()

        color = 'tab:blue'
        ax1.set_xlabel('time')
        ax1.set_ylabel('error', color=color)
        ax1.plot(t, error[b], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:red'
        ax2.set_ylabel('det(Covariance matrix)', color=color)  # we already handled the x-label with ax1
        ax2.plot(t, det[b], color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        #plt.title("Baddie " + str(b+1) + "Error and variance against time")
        plt.title("Error and variance against time")
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.legend(loc='lower right')
    plt.show()