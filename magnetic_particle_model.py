import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from scipy.integrate import odeint
import itertools
import sys

# Constants
n_particles = 4 # Number of particles
velocity_scale = 3 # Determines the magnitude of each random walk
box_length = 5 # For periodic boundary conditions.
n_points  = 100

# Initialize arrays to store positions, velocities, and polarities of particles. Also initialise the static state analysis vars
x = np.zeros(n_particles)
y = np.zeros(n_particles)
vx = np.zeros(n_particles)
vy = np.zeros(n_particles)
polarity = np.zeros(n_particles) # Polarity encodes the charge too.
colors = np.zeros(n_particles, dtype=str)
x_stor, y_stor, polarity_stor = [], [], []

# Set initial positions, and polarities of particles
polarity_values = np.array(list(itertools.chain(*[range(-10, 0), range(1, 10+1)]))) # Possible polarity values. This allows us to randomise polarities such that they may sum to zero.

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
t_max = 2
IPs = 10 # Number of initial field line IPs

"""Initialisers for fieldline calcs"""
C = 1 # Constant allows for flipping of B field for negative pols.
t_vals = np.linspace(0, t_max, 10) # This refers to the timesteps/arclengths that will be integrated over in odeint
""" Random points on sphere, following the algorithm as per Wolfram.
We follow the final algorithm on the page - pick a random normal variable for each x, y and z and then normalise to r.
"""
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return 0.1*vec
# 0.1 for radius. Is this the correct way to normalise to 0.1?
Ix0, Iy0, Iz0 = sample_spherical(IPs)
final_particle = np.zeros(n_particles)

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
    
    def field_line(M, t, C):
        p, q, z = M
        dBxds, dByds, dBzds = 0, 0, 0
        for i in range(n_particles): # Contribution to B from each particle
            norm = np.sqrt((p-x[i])**2 + (q-y[i])**2+(z)**2)
            dBxds += ((p-x[i])/norm) * polarity[i]
            dByds += ((q-y[i])/norm) * polarity[i]
            dBzds += z/norm * polarity[i] # No particle movement on z plane means the only term is that of the ip
            """
            Attempt at first order approximation. There should be 8 terms: one where we've added box length to x but not y, one we've added it to y not x etc.
            dBxds, dByds += ((p-(x[i]-box_length))/NEW_norm) * polarity[i], ((q-(y[i]+box_length))/NEW_norm) * polarity[i]
            dBxds, dByds += ((p-(x[i]-box_length))/NEW_norm) * polarity[i], ((q-y[i])/NEW_norm) * polarity[i]
            dBxds, dByds += ((p-(x[i]-box_length))/NEW_norm) * polarity[i], ((q-(y[i]-box_length))/NEW_norm) * polarity[i]
            dBxds, dByds += ((p-x[i])/NEW_norm) * polarity[i], ((q-(y[i]+box_length))/NEW_norm) * polarity[i]
            dBxds, dByds += ((p-x[i])/NEW_norm) * polarity[i], ((q-(y[i]-box_length))/NEW_norm) * polarity[i]
            dBxds, dByds += ((p-(x[i]+box_length))/NEW_norm) * polarity[i], ((q-(y[i]+box_length))/NEW_norm) * polarity[i]
            dBxds, dByds += ((p-(x[i]+box_length))/NEW_norm) * polarity[i], ((q-y[i])/NEW_norm) * polarity[i]
            dBxds, dByds += ((p-(x[i]+box_length))/NEW_norm) * polarity[i], ((q-(y[i]-box_length))/NEW_norm) * polarity[i]
            """
            
        dBds = [C * dBxds, C * dByds, C * dBzds] # Constant to confirm right direction of B field solving
        return dBds
    
    I = 0 # Test particle
    
    if np.linalg.norm((x[I],y[I])) < box_length: # i.e no need to bother checking if not in a suitable length.
        for k in range(1,n_particles): # Do not want to include 0. How can we have this so that it checks for every particle except the chosen test particle?
            pos = (x[k], y[k])
            for j in range(IPs):
                B0 = (Ix0[j]+x[I], Iy0[j]+y[I], Iz0[j])
                if polarity[I] >= 0:
                    sol = odeint(field_line, B0, t_vals , args=(C,))
                else:
                    sol = odeint(field_line, B0, t_vals, args=(-C,))
                end = sol[j]
                end = end[0:2] # Only need x and y coords
                if np.linalg.norm(end-pos) < interaction_radius: # Seems reasonable
                    final_particle[k] += 1
                    # print("Added to {} !".format(k))
    else:
        sys.exit()
    results = [final_particle[a]/np.sum(final_particle) for a in range(n_particles)]
    # print(results)
    axis.set_xlim(-box_length, box_length) # Ensures the animation looks more natural.
    axis.set_ylim(-box_length, box_length)
    """Static analysis storage (may also need to store colours"""
    x_stor.extend(x),y_stor.extend(y), polarity_stor.extend(polarity) # This isn't really helpful.
    print("After a time step", final_particle)

# update(1, x, y, polarity)
# Create the animation object
anim = FuncAnimation(fig, update, frames=num_of_frames, fargs=(x,y,polarity), interval=10, repeat = False) #Code doesn't seem to want to run if I put repeat false/

#anim.save('myanimation.gif') 
# Currently the animation object does not terminate on its own.
# Show the plot
plt.show()

"""
Potential solution for "Static analysis":
Store information on x, y and z at each step in a separate array outside of the update function. Also store polarities.
We can then use def fieldline outside of the update function and preform the field line analysis. We can use matplotlib to visualise our chosen instance.
If we want to continue the simulation from a given steady state, just feed back into update function with appropriate x, y, polarities.
"""
print(x_stor,y_stor,polarity_stor)
"""Now pick the time you would like to preform the analysis. We go from 0 to t_max with intervals of 0.1; use that to calculate what ts you can calculate."""
t_index = 7
print(x_stor[t_index], y_stor[t_index], polarity_stor[t_index]) # Not working for the reaons anticipated.
