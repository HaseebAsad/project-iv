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
        "Modify charges so that they are randomised "
        for i in range(n_particles): # Initialise IPs
            self.x[i] = np.random.uniform(-box_length, box_length)
            self.y[i] = np.random.uniform(-box_length, box_length)
            self.polarity[i] = np.random.choice(polarity_values) # Randomly assign polarity values, 2 for magnitude.
            self.colors[i] = 'r' if self.polarity[i] >= 0 else 'b'
        total_polarity = np.sum(self.polarity) # Calculate the total polarity
        # Check to ensure total pol is 0
        while total_polarity != 0:
            # Select polarity value that is not in the polarity array
            new_polarity = np.random.choice([p for p in polarity_values if p not in self.polarity])
            # Choose random particle to add or subtract the new polarity value
            random_index = np.random.randint(0, n_particles)
            self.polarity[random_index] += new_polarity
            total_polarity = np.sum(self.polarity)
        self.final_particle = np.zeros(n_particles)
        # Create a figure and axis object for the plot
        self.fig, self.axis = plt.subplots()
        self.num_of_frames = int(t_max/dt)
        self.interaction_radius = 0.5

    def update(self):
        self.axis.clear()
        """Initialisers for B-field"""
        Bx, By = 0, 0
        xf = np.arange(-box_length,box_length,0.25)
        yf = np.arange(-box_length,box_length,0.25)
        # Meshgrid
        X, Y = np.meshgrid(xf,yf)

        """Main for loop that checks for particle interactions, updates velocity based on a normal random walk, implements the resulting displacement, calculates B-field
        etc. """
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                dist = np.sqrt((self.x[i]-self.x[j])**2+(self.y[i]-self.y[j])**2) 
                if dist <= interaction_radius:
                    # print('Interaction between {} and {}'.format(i, j))
                    self.polarity [i] += self.polarity[j] # i.e. particle i adopts the polarity of J
                    self.polarity [j] = 0 # The particle that has its polarity adopted is now no longer of any polarity.
            if self.polarity [i] == 0:
                self.x[i], self.y[i] = 100*box_length, 100*box_length # This will remove the particle from physical view and no impact on magnetic field calculations.
                # print("particle {} is dead".format(i))
            # Update the colour of the polarity
            self.colors[i] = 'r' if self.polarity[i] >= 0 else 'b'
            """Update Velocities based on a normal random walk"""
            self.vx[i] = np.random.normal(loc=0, scale=velocity_scale) # Normal distribution centred around mean=loc and scaled by velocity_scale
            self.vy[i] = np.random.normal(loc=0, scale=velocity_scale)
            """Update particle positions, and implement the periodic boundary conditions."""
            norm = np.linalg.norm((self.x[i],self.y[i]))
            if norm < 3*box_length:
                self.x[i] += self.vx[i] * dt
                self.y[i] += self.vy[i] * dt
                if self.x[i] > box_length:
                    self.x[i] += -2*box_length
                if self.x[i] < -box_length:
                    self.x[i] += 2*box_length
                if self.y[i] > box_length:
                    self.y[i] += -2*box_length
                if self.y[i] < -box_length:
                    self.y[i] += 2*box_length
            """Calculate B-field, and add visual tags for each particle"""
            denom = np.sqrt((X-self.x[i])**2 + (Y-self.y[i])**2)
            Bx += ((X-self.x[i])/denom) * self.polarity[i]
            By += ((Y-self.y[i])/denom) * self.polarity[i]
            self.axis.text(self.x[i], self.y[i], f"{i, self.polarity[i]}") # Show index and charge of particle alongside.
        self.axis.scatter(self.x, self.y, c=self.colors, s=np.abs(self.polarity)*100)
        # Streamline plot
        self.axis.streamplot(X,Y,Bx,By, density=1.4, linewidth=None, color='#A23BEC')
        self.axis.set_xlim(-box_length, box_length) # Ensures the animation looks more natural.
        self.axis.set_ylim(-box_length, box_length)


if __name__ == '__main__':
    # Constants
    n_particles = 5 # Number of particles
    velocity_scale = 3 # Determines the magnitude of each random walk
    box_length = 5
    n_points  = 100
    polarity_values = np.arange(-10,10,1) # Possible polarity values. Does not exclude zero.
    # Time step, simulation length
    dt = 0.1
    t_max = 10
    num_of_frames = int(t_max/dt)

    IPs = 10 # Number of initial field line IPs
    interaction_radius = 0.5
    # Initialisers for fieldline calcs
    C = 1 # Constant allows for flipping of B field for negative pols.
    t_vals = np.arange(0, t_max, dt)
    # Generate random angles
    angles = np.linspace(0, np.pi/2, IPs) # From (0,pi/2) for the altitude, from (0,2pi) for azimuthal (eventually)
    # Define radius
    radii = 0.1
    # Convert polar coordinates to cartesian coordinates
    Ix0 = radii * np.cos(angles)
    Iy0 = radii * np.sin(angles)
    test_object = magnetic_particle()
    anim = FuncAnimation(test_object.fig, test_object.update, frames=num_of_frames, interval=10, repeat = False)
    plt.show()