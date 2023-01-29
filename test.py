import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.integrate import odeint

class magnetic_particle:
    def __init__(self):
        """Initialise parameters"""
        self.x = np.zeros(n_particles)
        self.y = np.zeros(n_particles)
        self.vx = np.zeros(n_particles)
        self.vy = np.zeros(n_particles)
        self.polarity = np.zeros(n_particles) # Polarity encodes the charge too.
        self.colors = np.zeros(n_particles, dtype=str)
    
    # Set initial positions, and polarities of particles

    def initial_charges(self):
        for i in range(n_particles): # Initialise IPs
            self.x[i] = np.random.uniform(-box_length, box_length)
            self.y[i] = np.random.uniform(-box_length, box_length)
            self.polarity[i] = np.random.choice(polarity_values) # Randomly assign polarity values, 2 for magnitude.
            self.colors[i] = 'r' if self.polarity[i] >= 0 else 'b'
        total_polarity = np.sum(self.polarity) # Calculate the total polarity
        # Check to ensure total pol is 0
        while total_polarity != 0:
            # Select polarity value that is not in the polarity array
            new_polarity = np.random.choice([p for p in polarity_values if p not in polarity])
            # Choose random particle to add or subtract the new polarity value
            random_index = np.random.randint(0, n_particles)
            self.polarity[random_index] += new_polarity
            total_polarity = np.sum(self.polarity)

if __name__ == '__main__':
    # Constants
    n_particles = 5 # Number of particles
    velocity_scale = 3 # Determines the magnitude of each random walk
    box_length = 5
    n_points  = 100
    polarity_values = np.arange(-10,10,1) # Possible polarity values. This allows us to randomise polarities such that they may sum to zero.
    

    test_object = magnetic_particle()
    print(test_object.polarity)
    """
    Call object/update fn
    """