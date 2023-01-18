import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import numpy as np
from scipy import integrate
import random

"""
First, find the field lines using the scipy inbuilt library. Field lines are the contours of the Flux function.
A field line is a curve that is everywhere tangent to B
so satisfies dx/ds=B(x(s))/|B(x(s))|, where s represents the arclength along the field line.
"""

"""
Worth noting that the matplotlib function will expect the inputs to be in Cartesian. So need to convert into Cartesian coordinates first.
Let our example be for a "single source". Priest 03
"""
"""
Example is an extremely basic flux function.
To determine the field lines of an arbitrary magnetic field requires you to write this code again.
Could create a class to prevent having to do this in the future with many more field lines?
"""

def fieldLine(M, t, C):
    x, y, z = M
    dBds = [-(x+2)/np.sqrt((x+2)**2+(y)**2+z**2)+ 2*(x-1)/np.sqrt((x-1)**2+(y-2)**2+z**2)- (x-2)/np.sqrt((x-2)**2+(y-1)**2+z**2) \
        , -(y)/np.sqrt((x+2)**2+(y)**2+z**2)+ 2*(y-2)/np.sqrt((x-1)**2+(y-2)**2+z**2)- (y-1)/np.sqrt((x-2)**2+(y-1)**2+z**2) \
        , -(z)/np.sqrt((x+2)**2+(y)**2+z**2)+ 2*(z)/np.sqrt((x-1)**2+(y-2)**2+z**2)- (z)/np.sqrt((x-2)**2+(y-1)**2+z**2)]
    return dBds

C = 1

t = np.linspace(0, 2, 100)
t2 = np.linspace(0,-2,100)
# Plot of the field lines in 2d using streamplot() https://www.geeksforgeeks.org/how-to-plot-a-simple-vector-field-in-matplotlib/ is a GREAT source (last example)
# Can ignore Z terms in a 2d projection.
# 1D arrays
x = np.arange(-5,5,0.1)
y = np.arange(-5,5,0.1)
  
# Meshgrid
X,Y = np.meshgrid(x,y)
  
# Assign vector directions
Ex = -(X+2)/np.sqrt((X+2)**2+(Y)**2)+ 2*(X-1)/np.sqrt((X-1)**2+(Y-2)**2)- (X-2)/np.sqrt((X-2)**2+(Y-1)**2)
Ey = -(Y)/np.sqrt((X+2)**2+(Y)**2)+ 2*(Y-2)/np.sqrt((X-1)**2+(Y-2)**2)- (Y-1)/np.sqrt((X-2)**2+(Y-1)**2)
  
# Depict illustration
plt.figure(figsize=(10, 10))
plt.streamplot(X,Y,Ex,Ey, density=1.4, linewidth=None, color='#A23BEC')
plt.plot(-2,0,'-or')
plt.plot(1,2,'-og')
plt.plot(2,1,'-or')
plt.title('Electromagnetic Field')
  
# Show plot with grid
plt.grid()
plt.show()

# Initial positions for field lines
ax = plt.axes(projection='3d')
N = 50 #Number of (equally spaced) ICs in the interval
l = 5 #half-length of volume of cube to plot in (symmetric around axis)

"""
Creating a 'box' of initial conditions around the null point of side lengths 0.1.
"""
for i in range (N):
    B0 = [-l + i*(2*l)/N, 0.5, 0] #equally spaced intervals in square/circle around (0,0)
    sol = integrate.odeint(fieldLine, B0, t, args=(C,)) 
    ax.plot3D(sol[:, 0],sol[:, 1],sol[:, 2])
for i in range (N):
    B0 = [0.5, -l + i*(2*l)/N, 0] #equally spaced intervals in square/circle around (0,0)
    sol = integrate.odeint(fieldLine, B0, t, args=(C,)) 
    ax.plot3D(sol[:, 0],sol[:, 1],sol[:, 2])
for i in range (N):
    B0 = [-l + i*(2*l)/N, 0.5, 0] #TODO: starting positions around sphere or random.random() but in full range (less preferred)
    sol = integrate.odeint(fieldLine, B0, t2, args=(C,)) 
    ax.plot3D(sol[:, 0],sol[:, 1],sol[:, 2])
for i in range (N):
    B0 = [0.5, -l + i*(2*l)/N, 0] #equally spaced intervals in square/circle around (0,0)
    sol = integrate.odeint(fieldLine, B0, t2, args=(C,)) 
    ax.plot3D(sol[:, 0],sol[:, 1],sol[:, 2])

ax.set_zlim(-5,5)
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
plt.show()