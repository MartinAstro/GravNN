import tempfile

import numpy as np

from GravNN.Regression.SHRegression import SHRegression
from GravNN.Regression.utils import (
    format_coefficients,
    populate_removed_degrees,
    save,
)


def test_setup(max_true_degree, regress_degree, remove_degree, initial_batch):
    import matplotlib.pyplot as plt

    from GravNN.CelestialBodies.Planets import Earth
    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
    from GravNN.Trajectories import DHGridDist

    planet = Earth()

    MAX_TRUE_DEG = max_true_degree
    REGRESS_DEG = regress_degree
    REMOVE_DEG = remove_degree

    def compute_dimensionality(N, M):
        return (N + 2) * (N + 1) - (M + 2) * (M + 1)

    dim = compute_dimensionality(REGRESS_DEG, REMOVE_DEG)

    sh_EGM2008 = SphericalHarmonics(planet.sh_file, REGRESS_DEG)

    trajectory = DHGridDist(planet, sh_EGM2008.radEquator, 90)
    # trajectory = RandomDist(planet, [planet.radius, planet.radius + 420000], 100000)

    x, a, u = get_sh_data(
        trajectory,
        planet.sh_file,
        max_deg=MAX_TRUE_DEG,
        deg_removed=REMOVE_DEG,
    )

    x0 = np.zeros((dim,))
    x0[0] = 0.0 if REMOVE_DEG != -1 else 1.0

    # Initialize the regressor
    regressor = SHRegression(
        REGRESS_DEG,
        REMOVE_DEG,
        planet.radius,
        planet.mu,
        kaula_factor=1e-3,
        max_batch_size=100,
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

    C_lm_error = (C_lm_true - C_lm) / C_lm_true * 100
    S_lm_error = (S_lm_true - S_lm) / S_lm_true * 100

    C_lm_error[np.isinf(C_lm_error)] = np.nan
    S_lm_error[np.isinf(S_lm_error)] = np.nan

    C_lm_avg_error = np.nanmean(np.abs(C_lm_error))
    S_lm_avg_error = np.nanmean(np.abs(S_lm_error))

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
        a_error = (
            np.linalg.norm(accelerations - a, axis=1) / np.linalg.norm(a, axis=1) * 100
        )

        print(f"\n AVERAGE CLM ERROR: {C_lm_avg_error} \n")
        print(f"\n AVERAGE SLM ERROR: {S_lm_avg_error} \n")

        print(f"\n ACCELERATION ERROR: {np.mean(a_error)}")

    # plot_coef_history(regressor.x_hat_hist, regressor.P_hat_hist, sh_EGM2008, REMOVE_DEG, start_idx=0)

    plt.show()
    return np.mean(a_error)


def main():
    # a_error = test_setup(
    #     max_true_degree=10,
    #     regress_degree=5,
    #     remove_degree=-1,
    #     initial_batch=100,
    # )
    # assert np.isclose(a_error, 0.0014472137798693197)

    a_error = test_setup(
        max_true_degree=100,
        regress_degree=50,
        remove_degree=-1,
        initial_batch=1000,
    )
    assert np.isclose(a_error, 0.0015995795115804907)


if __name__ == "__main__":
    main()
