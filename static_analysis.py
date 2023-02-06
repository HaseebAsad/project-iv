import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import itertools
import sys
from magnetic_particle_model import update
""" Initialisers from previous program"""
# Constants
n_particles = 4 # Number of particles

# Initialize arrays to store positions, velocities, and polarities of particles. Also initialise the static state analysis vars
x = np.zeros(n_particles)
y = np.zeros(n_particles)
polarity = np.zeros(n_particles) # Polarity encodes the charge too.

"""Main part"""

def run_update(n,x,y,polarity):
    # x,y,polarity are intial inputs, and then we iterate over update n times.
    for i in range(n):
        x, y, polarity = update(0,x,y,polarity)
    return x, y, polarity

print(run_update(0,x,y,polarity))