import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import itertools
import sys

# Constants
n_particles = 4 # Number of particles
velocity_scale = 3 # Determines the magnitude of each random walk
box_length = 5 # For periodic boundary conditions.

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
dt = 0.5
t_max = 0.5
IPs = 10 # Number of initial field line IPs

"""Initialisers for fieldline calcs"""
C = 1 # Constant allows for flipping of B field for negative pols.
s_vals = np.linspace(0, 2*box_length, 50) # This refers to the timesteps/arclengths that will be integrated over in odeint. box_length so integrates over a reasonable sized arc.
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
        denom = (np.sqrt((X-x[i])**2 + (Y-y[i])**2))**3
        Bx += ((X-x[i])/denom) * polarity[i]
        By += ((Y-y[i])/denom) * polarity[i]
        axis.text(x[i], y[i], f"{i, polarity[i]}") # Show index and charge of particle alongside.
    axis.scatter(x, y, c=colors, s=np.abs(polarity)*100)
    """ 
    Calculate the form of the B-field and plot. This gives a better visualisation than solving the field_line using odeint.
    """
    # Streamline plot
    axis.streamplot(X,Y,Bx,By, density=1.8, linewidth=None, color='#A23BEC')
    def field_line(M, t, C):
        p, q, z = M
        dBxds, dByds, dBzds = 0, 0, 0
        for i in range(n_particles): # Contribution to B from each particle
            norm = (np.sqrt((p-x[i])**2 + (q-y[i])**2+(z)**2))**3 #Even if we take the cube root factor out, its still not working too well.
            dBxds += ((p-x[i])/norm) * polarity[i]
            dByds += ((q-y[i])/norm) * polarity[i]
            dBzds += z/norm * polarity[i] # No particle movement on z plane means the only term is that of the ip
            """ First order terms. Very slow. Could make it quicker by putting the radial drop off without re-referencing the variable, but we'll see.

            norm  = (np.sqrt((p-(x[i]-box_length))**2 + (q-(y[i]+box_length))**2+(z)**2))**3
            dBxds += ((p-(x[i]-box_length))/norm) * polarity[i]
            dByds += ((q-(y[i]+box_length))/norm) * polarity[i]

            norm  = (np.sqrt((p-(x[i]-box_length))**2 + (q-(y[i])**2+(z)**2)))**3
            dBxds += ((p-(x[i]-box_length))/norm) * polarity[i]
            dByds +=  ((q-y[i])/norm) * polarity[i]

            norm  = (np.sqrt((p-(x[i]-box_length))**2 + (q-(y[i]-box_length))**2+(z)**2))**3 # could be error here or one above
            dBxds += ((p-(x[i]-box_length))/norm) * polarity[i]
            dByds += ((q-(y[i]-box_length))/norm) * polarity[i]

            norm  = (np.sqrt((p-(x[i]))**2 + (q-(y[i]+box_length))**2+(z)**2))**3
            dBxds += ((p-x[i])/norm) * polarity[i]
            dByds += ((q-(y[i]+box_length))/norm) * polarity[i]

            norm  = (np.sqrt((p-(x[i]))**2 + (q-(y[i]-box_length))**2+(z)**2))**3
            dBxds += ((p-x[i])/norm) * polarity[i]
            dByds += ((q-(y[i]-box_length))/norm) * polarity[i]

            norm  = (np.sqrt((p-(x[i]+box_length))**2 + (q-(y[i]+box_length))**2+(z)**2))**3
            dBxds += ((p-(x[i]+box_length))/norm) * polarity[i]
            dByds += ((q-(y[i]+box_length))/norm) * polarity[i]

            norm  = (np.sqrt((p-(x[i]+box_length))**2 + (q-(y[i])**2+(z)**2)))**3
            dBxds += ((p-(x[i]+box_length))/norm) * polarity[i]
            dByds += ((q-y[i])/norm) * polarity[i]

            norm  = (np.sqrt((p-(x[i]+box_length))**2 + (q-(y[i]-box_length))**2+(z)**2))**3 # could be error here or one above
            dBxds += ((p-(x[i]+box_length))/norm) * polarity[i]
            dByds += ((q-(y[i]-box_length))/norm) * polarity[i] """
            
        dBds = [C * dBxds, C * dByds, C * dBzds] # Constant to confirm right direction of B field solving
        return dBds
    
    I = 0 # Test particle
    
    if np.linalg.norm((x[I], y[I])) < 2 * box_length:
        positions = np.stack((x[1:], y[1:]), axis=-1)
        for j in range(IPs):
            B0 = (Ix0[j]+x[I], Iy0[j]+y[I], Iz0[j])
            if polarity[I] >= 0:
                sol = odeint(field_line, B0, s_vals , args=(C,))
            else:
                sol = odeint(field_line, B0, s_vals, args=(-C,))
            ends = sol[:, :2]
            distances = np.linalg.norm(ends[:, np.newaxis, :] - positions, axis=-1)
            final_particle[1:] += np.count_nonzero(distances < interaction_radius, axis=0)

    else:
        print("Particle 0 is out of range")
        sys.exit()
    results = [final_particle[a]/np.sum(final_particle) for a in range(n_particles)]
    # print(results)
    axis.set_xlim(-box_length, box_length) # Ensures the animation looks more natural.
    axis.set_ylim(-box_length, box_length)
    print("After a time step", final_particle)
    return x, y, polarity

# update(1, x, y, polarity)
# Create the animation object
# Could put this anim and plt.show() in a function which will also allow us to do our static analysis.
anim = FuncAnimation(fig, update, frames=num_of_frames, fargs=(x,y,polarity), interval=10, repeat = False) 
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

