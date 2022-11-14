import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

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
Example is the ABC magnetic field ()
"""

def fieldLine(M, t, A, B, C):
    x, y, z = M
    dBds = [A*np.sin(z)+C*np.cos(y), B*np.sin(x)+A*np.cos(z), C*np.sin(y)+B*np.cos(x)]
    return dBds

A = 1
B = 1
C = 1

y0 = [np.pi, -np.pi, 0.0]
t = np.linspace(0, 10, 101)

sol = integrate.odeint(fieldLine, y0, t, args=(A, B, C)) #Will give an array of numerical solutions to the ODE.
#Plot the solution to the ODE
plt.plot(t, sol[:, 0], 'b', label='x(t)')
plt.plot(t, sol[:, 1], 'g', label='y(t)')
plt.plot(t, sol[:, 2], 'r', label='z(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()