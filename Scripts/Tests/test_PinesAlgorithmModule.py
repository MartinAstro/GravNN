import numpy as np

import GravNN
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.PinesAlgorithm import *
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics


def accuracy():
    gravityModel = SphericalHarmonics(
        GravNN.__path__[0] + "/Files/GravityModels/GGM03S.txt",
        10,
    )
    planet = Earth(gravityModel)

    position_x = np.array([planet.radius, 0, 0])
    position_y = np.array([0, planet.radius, 0])
    position_z = np.array([0, 0, planet.radius])

    degree = 1
    n1, n2, n1q, n2q = compute_n_matrices(degree)
    accelerations_x = compute_acc_parallel(
        position_x,
        degree,
        planet.mu,
        planet.radius,
        n1,
        n2,
        n1q,
        n2q,
        gravityModel.C_lm,
        gravityModel.S_lm,
    )
    accelerations_y = compute_acc_parallel(
        position_y,
        degree,
        planet.mu,
        planet.radius,
        n1,
        n2,
        n1q,
        n2q,
        gravityModel.C_lm,
        gravityModel.S_lm,
    )
    accelerations_z = compute_acc_parallel(
        position_z,
        degree,
        planet.mu,
        planet.radius,
        n1,
        n2,
        n1q,
        n2q,
        gravityModel.C_lm,
        gravityModel.S_lm,
    )
    assert np.allclose(accelerations_x, np.array([-9.798286700796908, 0.0, 0.0]))
    assert np.allclose(accelerations_y, np.array([0.0, -9.798286700796908, 0.0]))
    assert np.allclose(accelerations_z, np.array([0.0, 0.0, -9.798286700796908]))

    degree = 4
    n1, n2, n1q, n2q = compute_n_matrices(degree)
    accelerations_x = compute_acc_parallel(
        position_x,
        degree,
        planet.mu,
        planet.radius,
        n1,
        n2,
        n1q,
        n2q,
        gravityModel.C_lm,
        gravityModel.S_lm,
    )
    accelerations_y = compute_acc_parallel(
        position_y,
        degree,
        planet.mu,
        planet.radius,
        n1,
        n2,
        n1q,
        n2q,
        gravityModel.C_lm,
        gravityModel.S_lm,
    )
    accelerations_z = compute_acc_parallel(
        position_z,
        degree,
        planet.mu,
        planet.radius,
        n1,
        n2,
        n1q,
        n2q,
        gravityModel.C_lm,
        gravityModel.S_lm,
    )
    assert np.allclose(
        accelerations_x,
        np.array([-9.814248185896481, 3.496054085131794e-05, 0.00010649159906769288]),
    )
    assert np.allclose(
        accelerations_y,
        np.array([-0.0001783443985249206, -9.813966166581835, -3.721700725970816e-05]),
    )
    assert np.allclose(
        accelerations_z,
        np.array([7.908837227492863e-05, -2.8203542979164537e-05, -9.766641408111397]),
    )

    degree = 4
    positions_all = np.array(
        [planet.radius, 0, 0, 0, planet.radius, 0, 0, 0, planet.radius],
    )
    n1, n2, n1q, n2q = compute_n_matrices(degree)
    accelerations_all = compute_acc_parallel(
        positions_all,
        degree,
        planet.mu,
        planet.radius,
        n1,
        n2,
        n1q,
        n2q,
        gravityModel.C_lm,
        gravityModel.S_lm,
    )
    assert np.allclose(
        accelerations_all[0:3],
        np.array([-9.814248185896481, 3.496054085131794e-05, 0.00010649159906769288]),
    )
    assert np.allclose(
        accelerations_all[3:6],
        np.array([-0.0001783443985249206, -9.813966166581835, -3.721700725970816e-05]),
    )
    assert np.allclose(
        accelerations_all[6:9],
        np.array([7.908837227492863e-05, -2.8203542979164537e-05, -9.766641408111397]),
    )

    print("Passed!")


if __name__ == "__main__":
    accuracy()
