import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import numpy as np
import math
from scipy import integrate
import random

"""
The following is a class that takes in two magnetic sources, and calculates useful numerical quantities from them, namely the heating flux.
The parameters that will be initialised are the respective fluxes and magnetic field strengths, the approach velocity, the impact parameter, initial positions of sources.
delta, the initial distance and angle will be calculated from their initial positions. The change in time (i.e. the duration of the process) and the alfven speed - canonical values used
but can add functionality to determine Alfven speed?
"""


class binaryReconnection:
    def __init__(self, F, B, v0, b, x, y, deltaT, va0):
        self.F = F
        self.B = B
        self.v0 = v0
        self.b = b
        self.x = x
        self.y = y
        self.deltaT = deltaT
        self.va0 = va0
    
    def initialDistance(self):
        self.delta = math.dist(self.x, self.y)
    
    #To define the field line plot properly, we can use the analysis mentioned early in Priest 04.
    #We assume the fields will always take the form given in Priest 04, so that they exhibit rotation.
    #We can either use the 2d Reconnection.py or fieldLine.py to plot the lines?
    def fieldLine(self, M, t, e):
        fieldStrength = abs(self.B[0]) #only plot one field to start with, test case
        x, y, z = M # is this necessary?
        dBds = [0, 0, e/(x**2+y**2+z**2)]
        return dBds
    
    def fieldLinePlot(self):
        t = np.linspace(0, 2, 100)
        t2 = np.linspace(0,-2,100)
        ax = plt.axes(projection='3d')
        N = 20 #Number of (equally spaced) ICs in the interval
        l = 5 #half-length of volume of cube to plot in (symmetric around axis)


    def angleDelta(self):
        self.angle = 2*np.arccos(self.b/self.delta) #check this
    
    #Input i to find the self-helicity for source i
    def energySelfHelicity(self, i, Leff):
        selfHelicity = (self.angle/np.pi)*self.F[i]**2 #check the usage of the angle here. Is it the right one?
        energisationParameter = (self.va0*self.deltaT)/(2*Leff)
        if energisationParameter > 1:
            print("The system exhibits quasi-static energisation")
            energy = selfHelicity/(24*np.pi*mu)*(self.angle/Leff) #again, careful over usage of self.angle instead of phi
        elif energisationParameter < 1:
            print("The system exhibits impulsive energisation")
            energy = energySelfHelicity(self.F, self.angle, self.deltaT, self.va0) # redo this line
        return energy

    def meanEnergyRelease(self, Nopp):
        Fmin = np.min([self.F[0],self.F[1]])
        self.releaseRate = (1/(3*np.pi*mu))*(Fmin**2)*Nopp*self.v0
        return releaseRate

    def heatFlux(self, N, Nopp):
        energyReleaseRate = self.releaseRate
        heatFlux = N*energyReleaseRate
        return heatFlux


if __name__ == "__main__":
    print("main")
    #Should Alfven speed go here or inside the class? What is more appropriate? It likely should go into the class as it varies.
    mu = 1.25663706e-6 #Permeability of free space/Vacuum. The magnetic constant is the magnetic permeability in a classical vacuum.