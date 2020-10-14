
import os, sys
import numpy as np
import GravNN

from GravNN.GravityModels.PinesAlgorithm import PinesAlgorithm
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics

def accuracy():
    gravityModel = SphericalHarmonics(GravNN.__path__[0]  + "/Files/GravityModels/GGM03S.txt", 10)
    planet = Earth(gravityModel)

    position_x = [planet.radius, 0, 0]
    position_y = [0, planet.radius, 0]
    position_z = [0, 0, planet.radius]

    degree = 1
    pines = PinesAlgorithm(planet.radius, planet.mu, degree, gravityModel.C_lm, gravityModel.S_lm)
    accelerations_x = pines.compute_acc(position_x)
    accelerations_y = pines.compute_acc(position_y)
    accelerations_z = pines.compute_acc(position_z)
    assert(np.allclose(accelerations_x, np.array([-9.798286700796908, 0.0, 0.0])))
    assert(np.allclose(accelerations_y, np.array([0.0, -9.798286700796908, 0.0])))
    assert(np.allclose(accelerations_z, np.array([0.0, 0.0, -9.798286700796908])))

    degree = 4
    pines = PinesAlgorithm(planet.radius, planet.mu, degree,gravityModel.C_lm, gravityModel.S_lm)
    accelerations_x = pines.compute_acc(position_x)
    accelerations_y = pines.compute_acc(position_y)
    accelerations_z = pines.compute_acc(position_z)
    assert(np.allclose(accelerations_x, np.array([-9.814248185896481, 3.496054085131794e-05, 0.00010649159906769288])))
    assert(np.allclose(accelerations_y, np.array([-0.0001783443985249206, -9.813966166581835, -3.721700725970816e-05])))
    assert(np.allclose(accelerations_z, np.array([7.908837227492863e-05, -2.8203542979164537e-05, -9.766641408111397])))


    degree = 4
    pines = PinesAlgorithm(planet.radius, planet.mu, degree,gravityModel.C_lm, gravityModel.S_lm)
    accelerations_all = pines.compute_acc([planet.radius, 0, 0, 0, planet.radius, 0, 0, 0, planet.radius])
    assert(np.allclose(accelerations_all[0:3], np.array([-9.814248185896481, 3.496054085131794e-05, 0.00010649159906769288])))
    assert(np.allclose(accelerations_all[3:6], np.array([-0.0001783443985249206, -9.813966166581835, -3.721700725970816e-05])))
    assert(np.allclose(accelerations_all[6:9], np.array([7.908837227492863e-05, -2.8203542979164537e-05, -9.766641408111397])))

    print("Passed!")

if __name__ == "__main__":
    accuracy()
