
import numpy as np
from numpy.random import seed

from Trajectories.UniformDist import UniformDist
from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics

import os, sys
from build.PinesAlgorithm import PinesAlgorithm
from build.Regression import Regression
from support.transformations import cart2sph, sphere2cart

def removeA0(positions, accelerations, mu):
    for i in range(int(len(positions)/3)):
        x, y, z = positions[3*i: 3*i+3]
        ax, ay, az = accelerations[3*i: 3*i+3]
        posSphere = cart2sph([[x,y,z]])
        r, theta, phi = posSphere[0,:]
        theta -= 180
        a0 = -mu/r**2
        sphereCoord = [[a0, theta, phi]]
        posCart = sphere2cart(sphereCoord)
        a0x, a0y, a0z = posCart[0,:]
        accelerations[3*i + 0] = accelerations[3*i + 0] - a0x
        accelerations[3*i + 1] = accelerations[3*i + 1] - a0y
        accelerations[3*i + 2] = accelerations[3*i + 2] - a0z
    return


def main():
    degree = 2
    planet = Earth()
    trajectory = UniformDist(planet, planet.radius, 100)
    gravityModel = SphericalHarmonics(os.path.dirname(os.path.realpath(__file__))  + "/../Files/GravityModels/GGM03S.txt", degree, trajectory=trajectory)

    accelerations = gravityModel.load()
    removeA0(trajectory.positions, accelerations, planet.mu)

    regression = Regression.Regression(trajectory.positions, accelerations, degree, planet.radius, planet.mu)
    regression.perform_regression()
    print(regression.coeff)

    #TODO: We need to save the results of the regression into gravity files. 


if __name__ == "__main__":
    main()