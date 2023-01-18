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
q = 1
p = 1
jt = sqrt(4*p**2+q**2)
jz = (1/2)*(q)

A = yy**2-xx**2 #from Parnell 1996 for 2D (not necessarily potential!) fields.
"""
We note that we can consider diffusion of a magnetic field with circular field lines
by consider the form of the flux function 
"""
eta = np.exp(-aa**2 - bb**2)
ux = -(1/2)*((jt-jz)*bb) 
uy = (1/2)*((jt+jz)/2)*aa 

"""
The above ux and uy are actually bx and by. What are ux and uy?
ux and uy must be defined elsewhere surely? Can find electric field as a result of B field
then find velocity using Ohm's law (for ideal fluids) E+u x B = 0. So made ux negative to account for this?
"""

plt.contour(xx,yy,A)
plt.contourf(aa,bb,eta)
plt.quiver(aa,bb,ux,uy)
# %%
