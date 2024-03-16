#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import itertools

# Constants
n_particles = 10 # Number of particles
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

# Check to ensure total flux is 0
while total_polarity != 0:
    # Select polarity value that is not in the polarity array
    new_polarity = np.random.choice([p for p in polarity_values if p not in polarity])
    # Choose random particle to add or subtract the new polarity value
    random_index = np.random.randint(0, n_particles)
    polarity[random_index] += new_polarity
    total_polarity = np.sum(polarity)

# Time step, simulation length
dt = 0 ## BE CAREFUL
t_max = 1
IPs = 1000 # Number of initial field line IPs

"""Initialisers for fieldline calcs"""
C = 1 # Constant allows for flipping of B field for negative pols.
s_vals = np.linspace(0, 2*box_length, 50) # This refers to the timesteps/arclengths that will be integrated over in odeint. box_length so integrates over a reasonable sized arc.
""" Random points on sphere, following the algorithm as per Wolfram.
We follow the final algorithm on the page - pick a random normal variable for each x, y and z and then normalise to r.
"""
def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0) # unit normalised.
    return 0.1*vec
Ix0, Iy0, Iz0 = sample_spherical(IPs)

# Create a figure and axis object for the plot
interaction_radius = 0.5

# Function to update the plot at each frame
def update(dt,x,y,polarity):
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
            print("particle {} is dead".format(i))
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
    return x, y, polarity

def run_update(x,y,polarity):
    final_particle = np.zeros(n_particles) # Because we're doing a static analysis, we would like this reset at each point.
    fig, axis = plt.subplots()
    x, y, polarity = update(0, x, y, polarity)
    # Clear the plot
    axis.clear()
    xf = np.arange(-box_length,box_length,0.25)
    yf = np.arange(-box_length,box_length,0.25)
    # Meshgrid
    X, Y = np.meshgrid(xf,yf)
    """Initialisers for B-field"""
    Bx, By = 0, 0
    for i in range(n_particles):
        denom = (np.sqrt((X-x[i])**2 + (Y-y[i])**2))**3
        Bx += ((X-x[i])/denom) * polarity[i]
        By += ((Y-y[i])/denom) * polarity[i]

        Bx += ((X-(x[i]-2*box_length))/((np.sqrt((X-(x[i]-2*box_length))**2 + (Y-(y[i]+2*box_length))**2))**3)) * polarity[i]
        By += ((Y-(y[i]+2*box_length))/((np.sqrt((X-(x[i]-2*box_length))**2 + (Y-(y[i]+2*box_length))**2))**3)) * polarity[i]

        Bx += ((X-(x[i]-2*box_length))/((np.sqrt((X-(x[i]-2*box_length))**2 + (Y-(y[i]))**2))**3)) * polarity[i]
        By += ((Y-(y[i]))/((np.sqrt((X-(x[i]-2*box_length))**2 + (Y-(y[i]))**2))**3)) * polarity[i]

        Bx += ((X-(x[i]-2*box_length))/((np.sqrt((X-(x[i]-2*box_length))**2 + (Y-(y[i]-2*box_length))**2))**3)) * polarity[i]
        By += ((Y-(y[i]-2*box_length))/((np.sqrt((X-(x[i]-2*box_length))**2 + (Y-(y[i]-2*box_length))**2))**3)) * polarity[i]


        Bx += ((X-(x[i]))/((np.sqrt((X-(x[i]))**2 + (Y-(y[i]+2*box_length))**2))**3)) * polarity[i]
        By += ((Y-(y[i]+2*box_length))/((np.sqrt((X-(x[i]))**2 + (Y-(y[i]+2*box_length))**2))**3)) * polarity[i]

        Bx += ((X-(x[i]))/((np.sqrt((X-(x[i]))**2 + (Y-(y[i]-2*box_length))**2))**3)) * polarity[i]
        By += ((Y-(y[i]-2*box_length))/((np.sqrt((X-(x[i]))**2 + (Y-(y[i]-2*box_length))**2))**3)) * polarity[i]


        Bx += ((X-(x[i]+2*box_length))/((np.sqrt((X-(x[i]+2*box_length))**2 + (Y-(y[i]+2*box_length))**2))**3)) * polarity[i]
        By += ((Y-(y[i]+2*box_length))/((np.sqrt((X-(x[i]+2*box_length))**2 + (Y-(y[i]+2*box_length))**2))**3)) * polarity[i]

        Bx += ((X-(x[i]+2*box_length))/((np.sqrt((X-(x[i]+2*box_length))**2 + (Y-(y[i]))**2))**3)) * polarity[i]
        By += ((Y-(y[i]))/((np.sqrt((X-(x[i]+2*box_length))**2 + (Y-(y[i]))**2))**3)) * polarity[i]

        Bx += ((X-(x[i]+2*box_length))/((np.sqrt((X-(x[i]+2*box_length))**2 + (Y-(y[i]-2*box_length))**2))**3)) * polarity[i]
        By += ((Y-(y[i]-2*box_length))/((np.sqrt((X-(x[i]+2*box_length))**2 + (Y-(y[i]-2*box_length))**2))**3)) * polarity[i]
        axis.text(x[i], y[i], f"{i, polarity[i]}") # Show index and charge of particle alongside.
    axis.scatter(x, y, c=colors, s=np.abs(polarity)*100)
    # Streamline plot
    axis.streamplot(X,Y,Bx,By, density=1.8, linewidth=None, color='#A23BEC')
    axis.set_xlim(-box_length, box_length) # Ensures the animation looks more natural.
    axis.set_ylim(-box_length, box_length)
    """
    Field line analysis:
    """
    def field_line(M, t, C):
        p, q, z = M
        dBxds, dByds, dBzds = 0, 0, 0
        for i in range(n_particles): # Contribution to B from each particle
            norm = (np.sqrt((p-x[i])**2 + (q-y[i])**2+(z)**2))**3 #Even if we take the cube root factor out, its still not working too well.
            dBxds += ((p-x[i])/norm) * polarity[i]
            dByds += ((q-y[i])/norm) * polarity[i]
            dBzds += z/norm * polarity[i] # No particle movement on z plane means the only term is that of the ip

            # Seem not to be as good at finding end points, likely due to the added "nulls" one gets by adding the first order periodic BCs

            dBxds += ((p-(x[i]-2*box_length))/((np.sqrt((p-(x[i]-2*box_length))**2 + (q-(y[i]+2*box_length))**2))**3)) * polarity[i]
            dByds += ((q-(y[i]+2*box_length))/((np.sqrt((p-(x[i]-2*box_length))**2 + (q-(y[i]+2*box_length))**2))**3)) * polarity[i]

            dBxds += ((p-(x[i]-2*box_length))/((np.sqrt((p-(x[i]-2*box_length))**2 + (q-(y[i]))**2))**3)) * polarity[i]
            dByds += ((q-(y[i]))/((np.sqrt((p-(x[i]-2*box_length))**2 + (q-(y[i]))**2))**3)) * polarity[i]

            dBxds += ((p-(x[i]-2*box_length))/((np.sqrt((p-(x[i]-2*box_length))**2 + (q-(y[i]-2*box_length))**2))**3)) * polarity[i]
            dByds += ((q-(y[i]-2*box_length))/((np.sqrt((p-(x[i]-2*box_length))**2 + (q-(y[i]-2*box_length))**2))**3)) * polarity[i]


            dBxds += ((p-(x[i]))/((np.sqrt((p-(x[i]))**2 + (q-(y[i]+2*box_length))**2))**3)) * polarity[i]
            dByds += ((q-(y[i]+2*box_length))/((np.sqrt((p-(x[i]))**2 + (q-(y[i]+2*box_length))**2))**3)) * polarity[i]

            dBxds += ((p-(x[i]))/((np.sqrt((p-(x[i]))**2 + (q-(y[i]-2*box_length))**2))**3)) * polarity[i]
            dByds += ((q-(y[i]-2*box_length))/((np.sqrt((p-(x[i]))**2 + (q-(y[i]-2*box_length))**2))**3)) * polarity[i]


            dBxds += ((p-(x[i]+2*box_length))/((np.sqrt((p-(x[i]+2*box_length))**2 + (q-(y[i]+2*box_length))**2))**3)) * polarity[i]
            dByds += ((q-(y[i]+2*box_length))/((np.sqrt((p-(x[i]+2*box_length))**2 + (p-(y[i]+2*box_length))**2))**3)) * polarity[i]

            dBxds += ((p-(x[i]+2*box_length))/((np.sqrt((p-(x[i]+2*box_length))**2 + (q-(y[i]))**2))**3)) * polarity[i]
            dByds += ((q-(y[i]))/((np.sqrt((p-(x[i]+2*box_length))**2 + (p-(y[i]))**2))**3)) * polarity[i]

            dBxds += ((p-(x[i]+2*box_length))/((np.sqrt((p-(x[i]+2*box_length))**2 + (q-(y[i]-2*box_length))**2))**3)) * polarity[i]
            dByds += ((q-(y[i]-2*box_length))/((np.sqrt((p-(x[i]+2*box_length))**2 + (q-(y[i]-2*box_length))**2))**3)) * polarity[i]


            
        dBds = [C * dBxds, C * dByds, C * dBzds] # Constant to confirm right direction of B field solving
        return dBds
    positions = np.stack((x[1:], y[1:]), axis=-1)
    I=0
    for j in range(IPs):
        B0 = (Ix0[j]+x[I], Iy0[j]+y[I], Iz0[j])
        if polarity[I] >= 0:
            sol = odeint(field_line, B0, s_vals , args=(C,))
        else:
            sol = odeint(field_line, B0, s_vals, args=(-C,))
        ends = sol[:, :2]
        for k in range(50): #len of s_vals
            if ends[k,0] > box_length:
                ends[k,0] += -2*box_length
            if ends[k,0] < -box_length:
                ends[k,0] += 2*box_length
            if ends[k,1] > box_length:
                ends[k,1] += -2*box_length
            if ends[k,1] < -box_length:
                ends[k,1] += 2*box_length
        distances = np.linalg.norm(ends[:, np.newaxis, :] - positions, axis=-1)
        final_particle[1:] += np.count_nonzero(distances < 0.25, axis=0)
    plt.show()
    print(final_particle)
    return x, y, polarity

x = [-2.26130156, -2.71487122, -1.9408815, 3.26123605, -2.59257517, -2.36060665, 1.82566019, -0.4908215, -1.20208611, -3.59610458]
y = [3.57235013, -0.99226015, 3.14429181, -4.10514027, -2.64590556, 4.7930023, 3.5966167, -3.08457845, 4.95818256, -0.80632212]
polarity = [10, -24, 4, -2, -9, 7, 3, 6, 4, 1]
run_update(x,y,polarity)
# %%
