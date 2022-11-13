#%%
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

npoints = 100 #Gives us the level of detail of your choice

x = np.linspace(-100,100,npoints)
y = np.linspace(-100,100,npoints)
# Create 2D arrays from your 1D x and y:
xx,yy = np.meshgrid(x,y,indexing='ij')

a = np.linspace(-100,100,10)
b = np.linspace(-100,100,10)

aa,bb = np.meshgrid(a,b,indexing='ij')

# Define eta, ux, uy etc via:
q = 1.6e-19
p = 1.6e-19
jt = sqrt(4*p**2+q**2) #jt is only defined by p. Could let q=1 by change of units?
jz = (1/2)*(q)

A = (jt/4)*((jt-jz)*yy**2-(jt+jz)*xx**2) #from Parnell 1996 for 2D fields.
"""
We note that we can consider diffusion of a magnetic field with circular field lines
by consider the form of the flux function 
"""
eta = np.exp(-aa**2 - bb**2)
ux = ((jt-jz)/2)*bb #velocity in x. By Parnell, this is the partial derivative of the flux function in y
uy = ((jt-jz)/2)*aa #velocity in y

"""
Check velocities of plasma flow
"""

plt.contour(xx,yy,A)
plt.contourf(aa,bb,eta)
plt.quiver(aa,bb,ux,uy)
# %%
