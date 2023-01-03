import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import numpy as np
import math
from scipy import integrate
import random

"""
The purpose of this code is to generate N particles (randomly) in a box, with N+ with positive polarity, N- with negative polarity. At each time interval, a random walk is taken.
We integrate field lines at each step to see the flux between flux and its neighbouring particles.
In the previous function work we've done, they generalise in the paper with density of sources in their calculations. This is what we need to do with ours too.
"""
#First, visualise N particles in a box

limit_x = 20
limit_y = 20

N = 200 # Number of particles
frac = random.random() # (Random) fraction of particles to be given positive polarity
R = 1

def force_init(n):
    count = np.linspace(0, N-1, N)
    x = (count * 2) % limit_x + R
    y = (count * 2) / limit_x + R
    position = np.column_stack((x, y))
    return position

position = force_init(N)
velocity = np.random.randn(N, 2) #Monte Carlo velocity

l = 0

while np.amax(abs(velocity)) > 0.01:
    if l%15 == 0:
        x = position[:,0]
        y = position[:,1]
        plt.plot(x, y, 'r+')
        plt.show()
    """
    We want the above to have a random number n of positive particles, and N-n negative particles.
    """
    position += velocity
    velocity *= 0.995 # messing with this changes the number of iterations the code does? How can I change this?

    # make 3D arrays with repeated position vectors to form combinations
    # diff_i[i][j] = position[i]
    # diff_j[i][j] = position[j]
    # diff[i][j] = vector pointing from i to j
    # norm[i][j] = sqrt( diff[i][j]**2 )
    diff_i = np.repeat(position.reshape(1, N, 2), N, axis=1).reshape(N, N, 2)
    diff_j = np.repeat(position.reshape(1, N, 2), N, axis=0)
    diff = diff_j - diff_i
    norm = np.linalg.norm(diff, axis=2)

    # make norm upper triangular (excluding diagonal)
    # This prevents double counting the i,j and j,i pairs
    """
    Take care of particle collision code.
    """
    collided = np.triu(norm < R, k=1)

    for i, j in zip(*np.nonzero(collided)):
        # unit vector from i to j
        unit = diff[i][j] / norm[i][j]

        # flip their velocity along the axis given by `unit`
        # and reduce momentum on that axis by 10%
        velocity[i] -= 1.9 * np.dot(unit, velocity[i]) * unit
        velocity[j] -= 1.9 * np.dot(unit, velocity[j]) * unit

        # push particle j to be 1 unit from i
        position[j] += ( R - norm[i][j] ) * unit

    # Masks
    xmax = position[:, 0] > limit_x
    xmin = position[:, 0] < 0
    ymax = position[:, 1] > limit_y
    ymin = position[:, 1] < 0

    # flip velocity
    velocity[xmax | xmin, 0] *= -1
    velocity[ymax | ymin, 1] *= -1

    # Ensure motion stays within the box
    position[xmax, 0] = limit_x - 2 * R
    position[xmin, 0] = 2 * R
    position[ymax, 1] = limit_y - 2 * R
    position[ymin, 1] = 2 * R

    l += 1