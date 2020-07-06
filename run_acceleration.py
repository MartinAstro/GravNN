
import numpy as np
from numpy.random import seed

from Trajectories.UniformDist import UniformDist
from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics
from GravityModels.SphericalHarmonics import SphericalHarmonics

import os, sys
sys.path.append(os.path.dirname(__file__) + "/build/PinesAlgorithm/")
import PinesAlgorithm

def main():
    degree = 4
    gravityModel = SphericalHarmonics(os.path.dirname(os.path.realpath(__file__))  + "/../Files/GravityModels/GGM03S.txt", 10)
    planet = Earth(gravityModel)

    trajectory = UniformDist(planet, planet.radius, 100)

    positions = np.reshape(trajectory.positions, (len(trajectory.positions)*3))
    pines = PinesAlgorithm.PinesAlgorithm(planet.radius, planet.mu, degree)
    accelerations = pines.compute_acc(positions, gravityModel.C_lm, gravityModel.S_lm)
    accelerations = np.reshape(np.array(accelerations), (int(len(np.array(accelerations))/3), 3))
    print(accelerations)




if __name__ == "__main__":
    main()