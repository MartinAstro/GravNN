
import numpy as np
from numpy.random import seed

from Trajectories.UniformDist import UniformDist
from CelestialBodies.Planets import Earth
from AccelerationAlgs.SphericalHarmonics import SphericalHarmonics

import os, sys
sys.path.append(os.path.dirname(__file__) + "/build/Regression/")
sys.path.append(os.path.dirname(__file__) + "/build/PinesAlgorithm/")

import PinesAlgorithm
import Regression

from support.transformations import cart2sph, sphere2cart

def removeA0(positions, accelerations, mu):
    for i in range(int(len(positions)/3)):
        x, y, z = positions[3*i: 3*i+3]
        ax, ay, az = accelerations[3*i: 3*i+3]
        posSphere = cart2sph([[x,y,z]])
        r, theta, phi = posSphere[0,:]
        theta = theta*np.pi/180.0
        phi = phi * np.pi/180.0
        a0 = mu/r**2
        a0x = a0*np.sin(theta)*np.cos(phi)
        a0y = a0*np.sin(theta)*np.sin(phi)
        a0z = a0*np.cos(theta)
        accelerations[3*i + 0] = accelerations[3*i + 0] - a0x
        accelerations[3*i + 1] = accelerations[3*i + 1] - a0y
        accelerations[3*i + 2] = accelerations[3*i + 2] - a0z
    return


def main():
    degree = 2
    planet = Earth()
    planet.loadSH()
    trajectory = UniformDist(planet, planet.geometry.radius, 100)

    positions = np.reshape(trajectory.positions, (len(trajectory.positions)*3))
    pines = PinesAlgorithm.PinesAlgorithm(planet.geometry.radius, planet.grav_info.mu, degree)
    accelerations = pines.compute_acc(positions, planet.grav_info.SH.C_lm, planet.grav_info.SH.S_lm)
    accelerations = np.array(accelerations)
    #removeA0(positions, accelerations, planet.grav_info.mu)

    regression = Regression.Regression(positions, accelerations, degree, planet.geometry.radius, planet.grav_info.mu)
    regression.perform_regression()
    print(regression.coeff)


if __name__ == "__main__":
    main()