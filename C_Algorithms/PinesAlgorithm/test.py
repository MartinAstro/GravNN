import os, sys
import numpy as np
sys.path.append(os.path.dirname(__file__) + "/../../")

from build.PinesAlgorithm import PinesAlgorithm
from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics

def main():
    degree = 2
    gravityModel = SphericalHarmonics(os.path.dirname(os.path.realpath(__file__))  + "/../../Files/GravityModels/GGM03S.txt", 10)
    planet = Earth(gravityModel)

    positions = [planet.radius, 0, 0]
    positions = [0, planet.radius, 0]

    pines = PinesAlgorithm.PinesAlgorithm(planet.radius, planet.mu, degree)
    accelerations = pines.compute_acc(positions, gravityModel.C_lm, gravityModel.S_lm)
    accelerations = np.reshape(np.array(accelerations), (int(len(np.array(accelerations))/3), 3))
    print(accelerations)


if __name__ == "__main__":
    main()