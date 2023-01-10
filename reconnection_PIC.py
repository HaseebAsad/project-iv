from boutdata.collect import collect
from boututils.datafile import DataFile
from math import sin, cos, pi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define the simulation parameters
nx = 256
ny = 256
lx = 1.0
ly = 1.0
cfl = 0.8

# Create the initial conditions for the simulation
Bx = [[0.0 for _ in range(ny)] for _ in range(nx)]
By = [[0.0 for _ in range(ny)] for _ in range(nx)]
Bz = [[0.0 for _ in range(ny)] for _ in range(nx)]
for i in range(nx):
    for j in range(ny):
        Bx1 = sin(2*pi*i*lx/nx)*cos(2*pi*j*ly/ny)
        By1 = -cos(2*pi*i*lx/nx)*sin(2*pi*j*ly/ny)
        Bx2 = sin(2*pi*i*lx/nx)*cos(2*pi*j*ly/ny + pi/2)
        By2 = -cos(2*pi*i*lx/nx)*sin(2*pi*j*ly/ny + pi/2)
        Bx[i][j] = Bx1 + Bx2
        By[i][j] = By1 + By2
        Bz[i][j] = 0.0

# Run the simulation for 10 time steps
dx = lx/nx
dy = ly/ny
dt = cfl*min(dx,dy) 
for t in range(10):
    # Time integration using Forward Euler Method
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            dBx_dt = (Bx[i+1][j]-Bx[i-1][j])/(2*dx) - (By[i][j+1]-By[i][j-1])/(2*dy)
            dBy_dt = (Bx[i][j+1]-Bx[i][j-1])/(2*dy) - (By[i+1][j]-By[i-1][j])/(2*dx)
            dBz_dt = (Bz[i+1][j]-Bz[i-1][j])/(2*dx) - (Bz[i][j+1]-Bz[i][j-1])/(2*dy)
            Bx[i][j] = Bx[i][j] + dBx_dt*dt
            By[i][j] = By[i][j] + dBy_dt*dt
            Bz[i][j] = Bz[i][j] + dBz_dt*dt

# When we need to analyse our given results, we can save them into a file as follows.
# df = DataFile("binary_reconnection.nc")
# df.write(Bx=Bx, By=By, Bz=Bz)

fig, ax = plt.subplots()

def update(num):
    ax.clear()
    x = range(256)
    y = range(256)
    plt.scatter(x,y,c=Bx[num][:], cmap='hot')
    ax.set_title('Bx at time step {}'.format(num))

ani = FuncAnimation(fig, update, frames=range(10), repeat=True)

plt.show()

