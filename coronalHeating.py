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

"""
In the following, we give the functions for the energetics as a result of the helicity injection.
Let B0 be the magnitude of the magnetic field (of which field?), mu =  permeability, v1= velocity of outward-propogating Alfvén wave.
"""
mu = 1.25663706e10-6 #permeability of free space

# The below is a definition of power including the radius between the pair of sources

def power(B0, va0, v1, phi, r, deltaT):
    omega = (phi/deltaT)
    P = B0**2/(3*mu*va0)*r^4*(omega**2)
    return P

# We can also define power in terms of the flux if we have a single source of the form B0=F/2*pi*r**2, check notes.

def fluxPower(F, va0, phi, deltaT):
    P = (F**2/(12*np.pi**2*mu))*(phi**2/(va0*deltaT))
    return P

#We can now define energy injection as the power times deltaT. We can also define it straight from helicity injection
def energy(B0, va0, v1, phi, r, deltaT):
    power = power(B0, va0, v1, phi, r, deltaT)
    return power*deltaT

# check equation 31 for self helicity injection
def energySelfHelicity(F, phi, deltaT, va0):
    selfHelicity = (phi/np.pi)*F**2
    energyInjection = (selfHelicity/(12*np.pi*mu))*(phi/(va0*deltaT))
    return energyInjection

# What kind of energisation do we have? Impulsive or quasi-static (often what appears in practice)? The next function determines the energy release based on the energisation parameter.

def energyRelease(F, phi, va0, Leff, deltaT):
    selfHelicity = (phi/np.pi)*F**2
    energisationParameter = (va0*deltaT)/(2*Leff)
    if energisationParameter > 1:
        print("The system exhibits quasi-static energisation")
        energy = selfHelicity/(24*np.pi*mu)*(phi/Leff)
    elif energisationParameter < 1:
        print("The system exhibits impulsive energisation")
        energy = energySelfHelicity(F, phi, deltaT, va0)
    return energy

"""
The following section of code refers to a BINARY COLLISION OF SOURCES WITH FLUXES F1 AND F2 RESPECTIVELY.
Let Delta be the INITIAL DISTANCE between the two sources.
Let b be their closest distance of approach
See figure 6b
"""


def angleDelta(delta, b):
    angle = 2*np.arccos(b/delta)
    return angle

def timeDelta(delta, b, v0):
    phi = angleDelta(delta,b)
    deltaT = (2*delta)/v0*np.sin(phi/2)
    return deltaT
 

"""
The source then goes on to show he we can rewrite the energisation parameter substiuting deltaT and our impact parameter. We find the the motion of magnetic sosurces is highly sub-Alfvenic,
and thus the energisation parameter is much greater than 1 - majority of interactions are quasi-static unless the impact parameter is about the same as the initial distance, in which case 
energisation parameter will be zero.
"""
#We can rewrite the energy release in terms of the impact parameter and Fmin, where Fmin is the minimum of the two fluxes involved in the interaction. This reduces the need to compute the angle delta.

def energyReleaseImpact(F1, F2, delta, b):
    Fmin = np.min([F1,F2])
    deltaE = (Fmin**2)/(6*np.pi**2*mu*b)*(np.arccos(b/delta))**2
    return deltaE

#We need the rate of collisions between the two sources to find the minimum rate of energy release. 

def collisionRate(v0, Nopp):
    nu = 2*v0*(Nopp/np.pi)**(1/2)
    return nu

#It is better to separately consider the mean rate of energy release without using the collission rate. Priest 04 offers an approximation which reduces our calculation to simply

def meanEnergyRelease(F1, F2, Nopp, v0):
    Fmin = np.min([F1,F2])
    releaseRate = (1/(3*np.pi*mu))*(Fmin**2)*Nopp*v0
    return releaseRate

#The corresponding heat flux from the majority species of density N therefore becomes

def heatFlux(F1, F2, N, Nopp, v0):
    energyReleaseRate = meanEnergyRelease(F1, F2, Nopp, v0)
    heatFlux = N*energyReleaseRate
    return heatFlux

#This is effectively all we need. We can also write the heat flux in terms of the mean flux densities

def heatFluxMagnetic(Bplus, Bminus, Fmin, Fmax, v0):
    eta = Fmin/Fmax #ratio of the minority to majority flux. What does this actually mean numerically?
    heatFlux = (1/(3*np.pi*mu))*eta*Bplus*Bminus*v0
    return heatFlux