import numpy as np

import GravNN
from GravNN.build.PinesAlgorithm import PinesAlgorithm
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.PinesAlgorithm import *
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics

degree = 175
gravityModel = SphericalHarmonics(
    GravNN.__path__[0] + "/Files/GravityModels/GGM03S.txt",
    degree,
)
planet = Earth(gravityModel)

N = 100
positions = np.random.uniform(1, 1.5, (3 * N)) * planet.radius

n1, n2, n1q, n2q = compute_n_matrices(degree)
acc = compute_acceleration(
    positions,
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
acc = compute_acc_parallel(
    positions,
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

pines = PinesAlgorithm.PinesAlgorithm(
    planet.radius,
    planet.mu,
    degree,
    gravityModel.C_lm,
    gravityModel.S_lm,
)


def timing():
    import timeit

    print(
        timeit.timeit(
            "compute_acceleration(positions, degree, planet.mu, planet.radius, n1, n2, n1q, n2q, gravityModel.C_lm, gravityModel.S_lm)",
            globals=globals(),
            number=100,
        ),
    )
    print(
        timeit.timeit(
            "compute_acc_parallel(positions, degree, planet.mu, planet.radius, n1, n2, n1q, n2q, gravityModel.C_lm, gravityModel.S_lm)",
            globals=globals(),
            number=100,
        ),
    )
    print(
        timeit.timeit(
            "pines.compute_acceleration(positions)",
            globals=globals(),
            number=100,
        ),
    )


if __name__ == "__main__":
    timing()
