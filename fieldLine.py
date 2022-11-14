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
Example is the ABC magnetic field 
"""

def fieldLine(M, t, A, B, C):
    x, y, z = M
    dBds = [A*np.sin(z)+C*np.cos(y), B*np.sin(x)+A*np.cos(z), C*np.sin(y)+B*np.cos(x)]
    return dBds

A = 1
B = np.sqrt(2/3)
C = np.sqrt(1/3)

B0 = [np.pi, -np.pi, 0.0] #ICs
t = np.linspace(-10, 10, 101)



sol = integrate.odeint(fieldLine, B0, t, args=(A, B, C)) #Will give an array of numerical solutions to the ODE.
#Plot the field lines with respect to their parameter (arclength)
plt.plot(t, sol[:, 0], 'b', label='x(t)')
plt.plot(t, sol[:, 1], 'g', label='y(t)')
plt.plot(t, sol[:, 2], 'r', label='z(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()

#Plot of the field lines (in 3D!)
ax = plt.axes(projection='3d')
for i in range (20):
    B0 = [random.random()*np.pi, random.random()*np.pi, random.random()*np.pi]
    sol = integrate.odeint(fieldLine, B0, t, args=(A, B, C))
    ax.plot3D(sol[:, 0],sol[:, 1],sol[:, 2])
plt.show()
"""
To get several field lines, as Yeates has done, we need several different ICs!
Could use a for loop to add all these lines in for plenty of different starting points.
"""