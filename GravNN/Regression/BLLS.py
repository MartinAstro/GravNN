import numpy as np

from GravNN.Regression.SHRegression import SHRegression
from GravNN.Regression.utils import format_coefficients, populate_removed_degrees, save


def iterate_lstsq(M, aVec, iterations, ridge_factor=None):
    if ridge_factor is not None:
        if isinstance(ridge_factor, np.ndarray):
            ridge_inv = np.linalg.inv(M.T @ M + ridge_factor)
        else:
            I = np.identity(M.shape[1])
            ridge_inv = np.linalg.inv(M.T @ M + ridge_factor * I)
        results = ridge_inv @ M.T @ aVec
    else:
        results = np.linalg.lstsq(M, aVec)[0]
    return results

    # results = np.linalg.lstsq(M, aVec)[0]
    # delta_a = aVec - np.dot(M, results)
    # for i in range(iterations):
    #     delta_coef = np.linalg.lstsq(M, delta_a)[0]
    #     results -= delta_coef
    #     delta_a = aVec - np.dot(M, results)


class BLLS:
    def __init__(
        self,
        max_deg,
        planet,
        remove_deg=-1,
        ridge_factor=None,
        kaula=True,
    ):
        self.N = max_deg  # Degree
        self.remove_deg = remove_deg
        self.SHRegressor = SHRegression(max_deg, remove_deg, planet.radius, planet.mu)
        self.ridge_factor = ridge_factor
        self.kaula = kaula

    def compute_ridge(self, inv_arg):
        # (Optional) Ridge Regression
        if self.ridge_factor is not None:
            if self.kaula:
                ridge = self.ridge_factor * self.SHRegressor.K_inv
            else:
                I = np.identity(inv_arg.shape[0])
                ridge = self.ridge_factor * I
        else:
            ridge = np.zeros(inv_arg.shape)
        return ridge

    def update(self, rVec, aVec, iterations=5):
        self.rVec1D = rVec.reshape((-1,))
        self.aVec1D = aVec.reshape((-1,))
        self.P = len(self.rVec1D)

        M = self.SHRegressor.populate_M(
            self.rVec1D,
            self.remove_deg,
        )

        inv_arg = M.T @ M
        self.ridge = self.compute_ridge(inv_arg)

        # Compute the Least Squares Solution
        results = np.linalg.inv(inv_arg + self.ridge) @ M.T @ self.aVec1D
        return results


class BLLS_PM:
    def __init__(self, max_deg, planet, remove_deg=-1):
        self.N = max_deg  # Degree
        self.a = planet.radius
        self.mu = planet.mu
        self.remove_deg = remove_deg

    def update(self, rVec, aVec, iterations=5):
        r = np.linalg.norm(rVec, axis=1)
        r_hat = rVec / r.reshape((-1, 1))
        a_pm = -self.mu * r_hat / r.reshape((-1, 1)) ** 2
        da_dmu = a_pm / self.mu
        M = da_dmu.reshape((-1, 1))

        self.rVec1D = rVec.reshape((-1,))
        self.aVec1D = aVec.reshape((-1,))
        self.P = len(self.rVec1D)

        results = iterate_lstsq(M, self.aVec1D, iterations)
        results /= self.mu
        return results


def test_setup(max_true_degree, regress_degree, remove_degree, ridge_factor=None):
    import tempfile
    import time

    from GravNN.CelestialBodies.Planets import Earth
    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
    from GravNN.Trajectories import DHGridDist

    MAX_TRUE_DEG = max_true_degree
    REGRESS_DEG = regress_degree
    REMOVE_DEG = remove_degree

    planet = Earth()
    sh_EGM2008 = SphericalHarmonics(planet.sh_file, REGRESS_DEG)

    trajectory = DHGridDist(planet, sh_EGM2008.radEquator, 10)

    x, a, u = get_sh_data(
        trajectory,
        planet.sh_file,
        max_deg=MAX_TRUE_DEG,
        deg_removed=REMOVE_DEG,
    )

    regressor = BLLS(
        REGRESS_DEG,
        planet,
        REMOVE_DEG,
        ridge_factor=ridge_factor,
        kaula=True,
    )
    start = time.time()
    results = regressor.update(x, a)
    C_lm, S_lm = format_coefficients(results, REGRESS_DEG, REMOVE_DEG)
    C_lm, S_lm = populate_removed_degrees(
        C_lm,
        S_lm,
        sh_EGM2008.C_lm,
        sh_EGM2008.S_lm,
        REMOVE_DEG,
    )

    print(time.time() - start)

    k = len(C_lm)
    C_lm_true = sh_EGM2008.C_lm[:k, :k]
    S_lm_true = sh_EGM2008.S_lm[:k, :k]

    C_lm_error = (C_lm_true - C_lm) / C_lm_true * 100
    S_lm_error = (S_lm_true - S_lm) / S_lm_true * 100

    C_lm_error[np.isinf(C_lm_error)] = np.nan
    S_lm_error[np.isinf(S_lm_error)] = np.nan

    C_lm_avg_error = np.nanmean(np.abs(C_lm_error))
    S_lm_avg_error = np.nanmean(np.abs(S_lm_error))

    # Save coefficents to temporary file
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
        error = (
            np.linalg.norm(accelerations - a, axis=1) / np.linalg.norm(a, axis=1) * 100
        )

    # Print Metrics
    # print("\nC_LM_REGRESS\n")
    # print(np.array2string(C_lm, precision=1))
    # print("\nC_LM_TRUE\n")
    # print(np.array2string(C_lm_true, precision=1))
    # print("\nC_LM_ERROR\n")
    # print(np.array2string(C_lm_error, precision=1))

    # print("\nS_LM_REGRESS\n")
    # print(np.array2string(S_lm, precision=1))
    # print("\nS_LM_TRUE\n")
    # print(np.array2string(S_lm_true, precision=1))
    # print("\nS_LM_ERROR\n")
    # print(np.array2string(S_lm_error, precision=1))

    # print(f"\n AVERAGE CLM ERROR: {C_lm_avg_error} \n")
    # print(f"\n AVERAGE SLM ERROR: {S_lm_avg_error} \n")

    print(f"\n ACCELERATION ERROR: {np.mean(error)}")

    return C_lm_avg_error, S_lm_avg_error, np.mean(error)


def main():
    # C_err, S_err = test_setup(
    #     max_true_degree=4,
    #     regress_degree=4,
    #     remove_degree=1,
    #     )
    # assert np.isclose(C_err, 1.207342327341248e-07)
    # assert np.isclose(S_err, 2.5174006765704146e-09)

    # C_err, S_err = test_setup(
    #     max_true_degree=4,
    #     regress_degree=4,
    #     remove_degree=-1,
    #     )
    # assert np.isclose(C_err, 1.6180533607033576e-06)
    # assert np.isclose(S_err, 4.275299330063149e-08)

    import matplotlib.pyplot as plt

    # deg_list = []
    # error_list = []
    # for degree in range(5, 50, 5):
    #     _, _, error = test_setup(
    #         max_true_degree=100,
    #         regress_degree=degree,
    #         remove_degree=-1,
    #         ridge_factor=1e-16,
    #     )
    #     deg_list.append(degree)
    #     error_list.append(error)
    # plt.plot(deg_list, error_list)

    plt.figure()

    deg_list = []
    error_list = []
    for ridge_factor in np.logspace(16, 26, 8, base=10):
        _, _, error = test_setup(
            max_true_degree=100,
            regress_degree=80,
            remove_degree=-1,
            ridge_factor=ridge_factor,
        )
        deg_list.append(ridge_factor)
        error_list.append(error)
    import matplotlib.pyplot as plt

    plt.semilogx(deg_list, error_list)
    plt.ylim([0, 5])
    plt.show()

    # This breaks!
    # test_setup(
    #     max_true_degree=100,
    #     regress_degree=90,
    #     remove_degree=-1,
    #     ridge_factor=1e10,
    # )

    # test_setup(
    #     max_true_degree=10,
    #     regress_degree=4,
    #     remove_degree=-1,
    # )


if __name__ == "__main__":
    main()
