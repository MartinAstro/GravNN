import os, sys
import numpy as np
sys.path.append(os.path.dirname(__file__) + "/../../")

from build.PinesAlgorithm import PinesAlgorithm
from CelestialBodies.Planets import Earth
from GravityModels.SphericalHarmonics import SphericalHarmonics

def main():
    gravityModel = SphericalHarmonics(os.path.dirname(os.path.realpath(__file__))  + "/../../Files/GravityModels/GGM03S.txt", 10)
    planet = Earth(gravityModel)

    position_x = [planet.radius, 0, 0]
    position_y = [0, planet.radius, 0]
    position_z = [0, 0, planet.radius]

    degree = 1
    pines = PinesAlgorithm.PinesAlgorithm(planet.radius, planet.mu, degree)
    accelerations_x = pines.compute_acc(position_x, gravityModel.C_lm, gravityModel.S_lm)
    accelerations_y = pines.compute_acc(position_y, gravityModel.C_lm, gravityModel.S_lm)
    accelerations_z = pines.compute_acc(position_z, gravityModel.C_lm, gravityModel.S_lm)
    assert(accelerations_x == (-9.798286700796908, 0.0, 0.0))
    assert(accelerations_y == (0.0, -9.798286700796908, 0.0))
    assert(accelerations_z == (0.0, 0.0, -9.798286700796908))

    degree = 4
    pines = PinesAlgorithm.PinesAlgorithm(planet.radius, planet.mu, degree)
    accelerations_x = pines.compute_acc(position_x, gravityModel.C_lm, gravityModel.S_lm)
    accelerations_y = pines.compute_acc(position_y, gravityModel.C_lm, gravityModel.S_lm)
    accelerations_z = pines.compute_acc(position_z, gravityModel.C_lm, gravityModel.S_lm)

    assert(accelerations_x == (-9.814248185896481, 3.496054085131794e-05, 0.00010649159906769288))
    assert(accelerations_y == (-0.0001783443985249206, -9.813966166581835, -3.721700725970816e-05))
    assert(accelerations_z == (7.908837227492863e-05, -2.8203542979164537e-05, -9.766641408111397))


if __name__ == "__main__":
    main()