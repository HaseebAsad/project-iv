import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.integrate import odeint
"""
Using our code previously built, this code will be a construction of small N particles interacting, modelling in a random walk motion.
We implement periodic boundary conditions: if particles leave a given domain, it will mirror itself and return back inside the "box" 
by flipping the coordinates (may be easier to simply implement taking away the length of the box.)
"""

# Constants
n_particles = 7 # Number of particles
velocity_scale = 3 # Determines the magnitude of each random walk
box_length = 5
n_points  = 100

# Initialize arrays to store positions, velocities, and polarities of particles
x = np.zeros(n_particles)
y = np.zeros(n_particles)
vx = np.zeros(n_particles)
vy = np.zeros(n_particles)
polarity = np.zeros(n_particles) # Polarity encodes the charge too.
colors = np.zeros(n_particles, dtype=str)

# Set initial positions, and polarities of particles
polarity_values = np.arange(-10,10,1) # Possible polarity values. This allows us to randomise polarities such that they may sum to zero.

for i in range(n_particles):
    x[i] = np.random.uniform(-box_length, box_length)
    y[i] = np.random.uniform(-box_length, box_length)
    polarity[i] = np.random.choice(polarity_values) # Randomly assign polarity values, 2 for magnitude.
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
t_max = 3
# Initialisers for fieldline calcs
C = 1 # Test constant.
t_vals = np.arange(0, t_max, dt)
# Generate random angles
angles = np.arange(0,np.pi/2, 50) # From (0,pi/2) for the altitude, from (0,2pi) for azimuthal (eventually)
# Define radius
radii = 0.1 
# Convert polar coordinates to cartesian coordinates
Ix0 = radii * np.cos(angles)
Iy0 = radii * np.sin(angles)
Ixy = np.column_stack((Ix0, Iy0)) # Initial field line starting positions.

# Create a figure and axis object for the plot
fig, axis = plt.subplots()
num_of_frames = int(t_max/dt)
interaction_radius = 0.5

# Function to update the plot at each frame
def update(t,x,y,polarity):
    # Clear the plot
    axis.clear()
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
            dist = np.sqrt((x[i]-x[j])**2+(y[i]-y[j])**2) 
            if dist <= interaction_radius:
                # print('Interaction between {} and {}'.format(i, j))
                polarity [i] += polarity[j] # i.e. particle i adopts the polarity of J
                polarity [j] = 0 # The particle that has its polarity adopted is now no longer of any polarity.
        if polarity [i] == 0:
            x[i], y[i] = 100*box_length, 100*box_length # This will remove the particle from physical view and no impact on magnetic field calculations.
            # print("particle {} is dead".format(i))
        # Update the colour of the polarity
        colors[i] = 'r' if polarity[i] >= 0 else 'b'
        """Update Velocities based on a normal random walk"""
        vx[i] = np.random.normal(loc=0, scale=velocity_scale) # Normal distribution centred around mean=loc and scaled by velocity_scale
        vy[i] = np.random.normal(loc=0, scale=velocity_scale)
        """Update particle positions, and implement the periodic boundary conditions."""
        norm = np.linalg.norm((x[i],y[i]))
        if norm < 3*box_length:
            x[i] += vx[i] * dt
            y[i] += vy[i] * dt
            if x[i] > box_length:
                x[i] += -2*box_length
            if x[i] < -box_length:
                x[i] += 2*box_length
            if y[i] > box_length:
                y[i] += -2*box_length
            if y[i] < -box_length:
                y[i] += 2*box_length
        """Calculate B-field, and add visual tags for each particle"""
        denom = np.sqrt((X-x[i])**2 + (Y-y[i])**2)
        Bx += ((X-x[i])/denom) * polarity[i]
        By += ((Y-y[i])/denom) * polarity[i]
        axis.text(x[i], y[i], f"{i, polarity[i]}") # Show index and charge of particle alongside.
    axis.scatter(x, y, c=colors, s=np.abs(polarity)*100)
    """ 
    Calculate the form of the B-field and plot. This gives a better visualisation than solving the field_line using odeint.
    """
    # Streamline plot
    axis.streamplot(X,Y,Bx,By, density=1.4, linewidth=None, color='#A23BEC')
    """
    Here we will use odeint to solve field lines. Steps are in Notion.
    1. Create a function that defines the function we are integrating.
    2. Integrate using odeint.
    3. Use the Ixy array for field line starting positions, translating by x[i],y[i] for chosen particle.
    """
    def field_line(M, t, C):
        p, q = M
        dBxds, dByds = 0, 0
        for i in range(n_particles): # Contribution to B from each particle
            norm = np.sqrt((p-x[i])**2 + (q-y[i])**2)
            dBxds += ((p-x[i])/norm) * polarity[i]
            dByds += ((q-y[i])/norm) * polarity[i]
        dBds = [dBxds, dByds]
        return dBds
    i = 3
    for j in range(len(Ixy)):
        B0 = (Ix0[j]+x[i], Iy0[j]+y[j])
        sol = odeint(field_line, B0, t_vals, args=(C,))
    print(sol)
    axis.set_xlim(-box_length, box_length) # Ensures the animation looks more natural.
    axis.set_ylim(-box_length, box_length)

# Create the animation object
anim = FuncAnimation(fig, update, frames=num_of_frames, fargs=(x,y,polarity), interval=10, repeat = False)
#anim.save('myanimation.gif') 
# Currently the animation object does not terminate on its own.
# Show the plot
plt.show()
