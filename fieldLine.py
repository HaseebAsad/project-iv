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
Below is a simple harmonic oscillator
"""

def fieldLine(B, t, k, m):
    x, y = B
    dBds = [-(k/m)*y, x]
    return dBds

k=0.5
m=1

y0 = [np.pi - 0.1, 0.0]
t = np.linspace(0, 10, 101)

sol = integrate.odeint(fieldLine, y0, t, args=(m, k)) #Will give an array of numerical solutions to the ODE.
#Plot the solution to the ODE
plt.plot(t, sol[:, 0], 'b', label='theta(t)')
plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()