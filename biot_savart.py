import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
This code calculates the magnetic field of n point sources at a given position in 3D space using
and animates the magnetic field lines based on that calculation. 
"""

# Set the parameters for the point sources
n = 3  # Number of point sources
q = np.array([1, -1, 1])  # Charge of each point source (in units of e). We can randomise this in further constructions.
r_0 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])  # Position of each point source

# The position where the magnetic field is to be calculated
r = np.array([0, 0, 0])

# Initialize the magnetic field array
B = np.zeros(3)

# Rudimentary approach to calculating B, a more realistic model would need to use numerical methods such as finite differences, Runge-Kutta etc.
# Calculate the magnetic field due to each point source
for i in range(n):
    r_rel = r - r_0[i]  # Position relative to point source
    r_rel_norm = np.linalg.norm(r_rel)  # Distance to point source
    B += (q[i] / (4 * np.pi) * (r_rel / r_rel_norm**3)) # Biot-Savart Law

print(B) 

# This function calculates B as above for the animation.
def calculate_B(r):
    # Initialize the magnetic field array
    B = np.zeros(3)

    # Calculate the magnetic field due to each point source
    for i in range(n):
        r_rel = r - r_0[i]  # Position relative to point source
        r_rel_norm = np.linalg.norm(r_rel)  # Distance to point source
        B += (q[i] / (4 * np.pi) * (r_rel / r_rel_norm**3)) # Biot-Savart Law
    return B

# Function to calculate the field line
def field_line(r_start):
    r = r_start # Number of frames to be animated, effectively.
    x = [r[0]]
    y = [r[1]]
    z = [r[2]]
    for i in range(n_points):
        B = calculate_B(r)
        r += dt * B / np.linalg.norm(B)
        x.append(r[0])
        y.append(r[1])
        z.append(r[2])
    return x,y,z

n_points = 10
dt = 0.1

fig = plt.figure()
ax = plt.axes(projection='3d')

# Function to animate the above.
def animate(i):
    x,y,z = field_line(r_start[i])
    ax.clear() # Clear plot, reset with updated plot.
    ax.plot(x, y, z)
    ax.set_title('Field line Animation')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# Initial positions to determine the field lines
n_pos = 100
r_start = np.random.rand(n_pos,3)

ani = animation.FuncAnimation(fig, animate, frames=len(r_start), repeat=True)
plt.show()

