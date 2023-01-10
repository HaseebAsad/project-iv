import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
n_particles = 1000  # Number of plasma particles
B = 1.0  # Magnetic field strength
q = 1.0  # Charge of plasma particles
m = 1.0  # Mass of plasma particles

# Initialize arrays to store the positions, velocities, and polarities of the particles
x = np.zeros(n_particles)
y = np.zeros(n_particles)
vx = np.zeros(n_particles)
vy = np.zeros(n_particles)
polarity = np.zeros(n_particles)
colors = np.zeros(n_particles, dtype=str)

# Set the initial positions, velocities, and polarities of the particles
for i in range(n_particles):
    x[i] = np.random.uniform(-1, 1)
    y[i] = np.random.uniform(-1, 1)
    vx[i] = np.random.uniform(-1, 1)
    vy[i] = np.random.uniform(-1, 1)
    polarity[i] = 1 if i < n_particles / 2 else -1
    colors[i] = 'r' if polarity[i] == 1 else 'b'

# Set the time step and simulation length
dt = 0.01
t_max = 1

# Create a figure and axis object for the plot
fig, axis = plt.subplots()

# Function to update the plot at each frame
def update(t):
    # Update the velocities of the particles based on the Lorentz force
    for i in range(n_particles):
        ax = polarity[i] * q * (vy[i] * B) / m
        ay = -polarity[i] * q * (vx[i] * B) / m
        vx[i] += ax * dt
        vy[i] += ay * dt

    # Update the positions of the particles based on their velocities
    for i in range(n_particles):
        x[i] += vx[i] * dt
        y[i] += vy[i] * dt

    # Clear the plot and update the positions of the particles
    axis.clear()
    axis.scatter(x, y, c=colors)

# Create the animation object
anim = FuncAnimation(fig, update, frames=np.arange(0, t_max, dt), interval=10)

# Show the plot
plt.show()
