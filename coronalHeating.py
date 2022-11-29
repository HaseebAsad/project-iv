import numpy as np

"""
The following program provides outputs of energy release, flux release and so on in the process of binary reconnection.
We can later adopt values typical of that observed at the sun to find the heating rate at the sun itself.
"""

"""
We estimate helicity injection (change in helicity) as a result of the relative motions of a pair of sources.
The general form of the helicity injection is below
"""

# In the following, wi should be an array of N, as should Fi, where N is the number of sources we are analysing.
# theta is the angle between a pair of sources, and should be length mod(N/2) in modulo 2. Angle in radians
def helicityInjection(wi,Fi,theta):
    selfHelicity, mutualHelicity = 0, 0
    for i in range(len(wi)):
        selfHelicity += wi[i]*(Fi[i]**2)
    for j in range(len(Fi)-1):
        mutualHelicity += theta[j]*Fi[j]*Fi[j+1] #this line needs a check for the indexing, will absolutely not work in generality.
    return -1/(2*np.pi)*(selfHelicity+mutualHelicity)

# note, if we choose angular velocity in unit time, angular velocity is just theta! This is a simple rotation of a source.
# print(helicityInjection([np.pi/4,np.pi/4],[5,-2],[np.pi/4]))
# If the sources are unbalanced, the above can be reduced to return -theta/np.pi*(F1+F2)*2. However, this does not allow us to analyse the selfHelicity and mutualHelicity individually.

