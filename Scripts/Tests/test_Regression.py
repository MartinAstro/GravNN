import tempfile

import matplotlib.pyplot as plt
import numpy as np

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Regression.SHRegression import SHRegression, SHRegressorSequential
from GravNN.Regression.utils import (
    format_coefficients,
    populate_removed_degrees,
    save,
)
from GravNN.Trajectories import DHGridDist


def eval_coef_error(C_lm, S_lm, C_lm_true, S_lm_true):
    C_lm_error = (C_lm_true - C_lm) / C_lm_true * 100
    S_lm_error = (S_lm_true - S_lm) / S_lm_true * 100

    C_lm_error[np.isinf(C_lm_error)] = np.nan
    S_lm_error[np.isinf(S_lm_error)] = np.nan

    C_lm_avg_error = np.nanmean(np.abs(C_lm_error))
    S_lm_avg_error = np.nanmean(np.abs(S_lm_error))
    return C_lm_avg_error, S_lm_avg_error


def test_setup(max_true_degree, regress_degree, remove_degree, sequential_params=None):
    planet = Earth()

    MAX_TRUE_DEG = max_true_degree
    REGRESS_DEG = regress_degree
    REMOVE_DEG = remove_degree

    sh_EGM2008 = SphericalHarmonics(planet.sh_file, REGRESS_DEG)

    trajectory = DHGridDist(planet, sh_EGM2008.radEquator, 90)
    # trajectory = RandomDist(planet, [planet.radius, planet.radius + 420000], 100000)

    x, a, u = get_sh_data(
        trajectory,
        planet.sh_file,
        max_deg=MAX_TRUE_DEG,
        deg_removed=REMOVE_DEG,
    )

    # Initialize the regressor
    if sequential_params is None:
        regressor = SHRegression(
            REGRESS_DEG,
            REMOVE_DEG,
            planet.radius,
            planet.mu,
            kaula_factor=1e3,
            max_batch_size=100,
        )
    else:
        SEQ_PARAMS = sequential_params
        regressor = SHRegressorSequential(
            REGRESS_DEG,
            SEQ_PARAMS,
            planet,
        )
    regressor.update(x, a)

    # Format the coefficients
    C_lm, S_lm = format_coefficients(regressor.x_hat, regressor.N, REMOVE_DEG)
    C_lm, S_lm = populate_removed_degrees(
        C_lm,
        S_lm,
        sh_EGM2008.C_lm,
        sh_EGM2008.S_lm,
        REMOVE_DEG,
    )

    # Compute Error + Metrics
    k = len(C_lm)
    C_lm_true = sh_EGM2008.C_lm[:k, :k]
    S_lm_true = sh_EGM2008.S_lm[:k, :k]

    C_lm_avg_error, S_lm_avg_error = eval_coef_error(C_lm, S_lm, C_lm_true, S_lm_true)
    print(f"\n AVERAGE CLM ERROR: {C_lm_avg_error} \n")
    print(f"\n AVERAGE SLM ERROR: {S_lm_avg_error} \n")

    with tempfile.NamedTemporaryFile() as tmpfile:
        save(tmpfile.name, planet, C_lm, S_lm)
        regressed_model = SphericalHarmonics(tmpfile.name, REGRESS_DEG)
        accelerations = regressed_model.compute_acceleration(trajectory.positions)

        x, a, u = get_sh_data(
            trajectory,
            planet.sh_file,
            max_deg=MAX_TRUE_DEG,
            deg_removed=-1,
        )
        da = np.linalg.norm(accelerations - a, axis=1)
        a_mag = np.linalg.norm(a, axis=1)
        a_error = da / a_mag * 100

        print(f"\n ACCELERATION ERROR: {np.mean(a_error)}")

    plt.show()
    return np.mean(a_error)


def main():
    # a_error = test_setup(
    #     max_true_degree=10,
    #     regress_degree=5,
    #     remove_degree=-1,
    # )
    # assert np.isclose(a_error, 0.0014472137798693197)

    # a_error = test_setup(
    #     max_true_degree=100,
    #     regress_degree=50,
    #     remove_degree=-1,
    # )
    # assert np.isclose(a_error, 0.0012637670031650306)

    test_setup(
        max_true_degree=100,
        regress_degree=10,
        remove_degree=-1,
        sequential_params=45,
    )


if __name__ == "__main__":
    main()
