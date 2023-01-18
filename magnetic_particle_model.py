import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

"""
Using our code previously built, this code will be a construction of small N particles interacting, modelling in a random walk motion.
We implement periodic boundary conditions: if particles leave a given domain, it will mirror itself and return back inside the "box" 
by flipping the coordinates (may be easier to simply implement taking away the length of the box.)
"""

# Constants
n_particles = 3  # Number of particles
velocity_scale = 20 # Determines the magnitude of each random walk
box_length = 50
n_points  = 100

# Initialize arrays to store positions, velocities, and polarities of particles
x = np.zeros(n_particles)
y = np.zeros(n_particles)
vx = np.zeros(n_particles)
vy = np.zeros(n_particles)
polarity = np.zeros(n_particles) # Polarity encodes the charge too.
B = np.zeros((n_particles, 3))
colors = np.zeros(n_particles, dtype=str)

# Set initial positions, and polarities of particles
polarity_values = [-5,-4,-3,-2,-1,1,2,3,4,5] # Possible polarity values. This allows us to randomise polarities such that they may sum to zero.

for i in range(n_particles):
    x[i] = np.random.uniform(-box_length, box_length)
    y[i] = np.random.uniform(-box_length, box_length)
    polarity[i] = np.random.choice(polarity_values) # Randomly assign polarity values
    colors[i] = 'r' if polarity[i] >= 0 else 'b'

total_polarity = np.sum(polarity) # Calculate the total polarity

# Check to ensure total pol is 0
while total_polarity != 0:
    # Select polarity value that is not in the polarity array
    new_polarity = np.random.choice([p for p in polarity_values if p not in polarity])
    # Choose random particle to add or subtract the new polarity value
    random_index = np.random.randint(0, n_particles)
    polarity[random_index] += new_polarity
    total_polarity = np.sum(polarity)

# Time step, simulation length
dt = 0.1
t_max = 1

# Create a figure and axis object for the plot
fig, axis = plt.subplots()
num_of_frames = int(t_max/dt)
interaction_radius = 5

# Function to update the plot at each frame
def update(t,x,y,polarity):

    # Update velocities of the particles based on a normal random walk
    # Could have code here to prevent the blocks of code running for particles that have None as their position. e.g. if x[i] != None:
    for i in range(n_particles):
        vx[i] = np.random.normal(loc=0, scale=velocity_scale) # Normal distribution centred around mean=loc and scaled by velocity_scale
        vy[i] = np.random.normal(loc=0, scale=velocity_scale)

    # Check for interactions between particles and update their properties accordingly.
    for i in range(n_particles):
        for j in range(i+1, n_particles):
            dist = np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2) 
            if dist <= interaction_radius:
                # print('Interaction between {} and {}'.format(i, j))
                polarity [i] += polarity[j] # i.e. particle i adopts the polarity of J
                polarity [j] = 0 # The particle that has its polarity adopted is now no longer of any polarity.
        if polarity [i] == 0:
            x[i], y[i] = 100*box_length, 100*box_length # This will remove the particle from physical view and no impact on magnetic field calculations.
            print("particle {} is dead".format(i))
        # Update the colour of the polarity
        colors[i] = 'r' if polarity[i] >= 0 else 'b'

    
    # Update the positions of the particles based on their velocities
    for i in range(n_particles):
        x[i] += vx[i] * dt
        y[i] += vy[i] * dt
        norm = np.sqrt(x[i]**2+y[i]**2)
        if norm < 3*box_length:
            if x[i] > box_length:
                x[i] += -2*box_length
            if x[i] < -box_length:
                x[i] += 2*box_length
            if y[i] > box_length:
                y[i] += -2*box_length
            if y[i] < -box_length:
                y[i] += 2*box_length

    """ 
    Calculate the form of the B-field. Will then need to solve them using odeint?
    """
    Bx, By = 0, 0
    xf = np.arange(-box_length,box_length,0.5)
    yf = np.arange(-box_length,box_length,0.1)
    # Meshgrid
    X, Y = np.meshgrid(xf,yf)

    for i in range(n_particles): # Contribution to B from each particle
        norm = np.sqrt((X-x[i])**2 + (Y-y[i])**2)
        Bx += ((X-x[i])/norm) * polarity[i]
        By += ((Y-y[i])/norm) * polarity[i]

    axis.streamplot(X,Y,Bx,By, density=1.4, linewidth=None, color='#A23BEC')

    # Clear the plot and update the positions of the particles
    axis.clear()
    axis.scatter(x, y, c=colors, s=np.abs(polarity)*100)
    axis.set_xlim(-box_length, box_length) # Ensures the animation looks more natural.
    axis.set_ylim(-box_length, box_length)
    for i in range(n_particles):
        axis.text(x[i], y[i], f"{i, polarity[i]}") # Show index and charge of particle alongside.

# Create the animation object
anim = FuncAnimation(fig, update, frames=num_of_frames, fargs=(x,y,polarity), interval=10)
# Currently the animation object does not terminate on its own.

# Show the plot
plt.show()
