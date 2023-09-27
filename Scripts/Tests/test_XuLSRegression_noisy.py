import matplotlib.pyplot as plt
import numpy as np
import sigfig

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Regression.utils import (
    RegressSolution,
)
from GravNN.Regression.XuLS import AnalyzeRegression, XuLS
from GravNN.Trajectories import RandomDist


def print_coefficients(C_lm, S_lm):
    for i in range(len(C_lm)):
        for j in range(i + 1):
            print(
                f"({i},{j}): {sigfig.round(float(C_lm[i,j]),sigfigs=2, notation='scientific')} \t {sigfig.round(float(S_lm[i,j]),sigfigs=2,notation='scientific')}",
            )


def compute_dbeta(C_lm_true, S_lm_true, C_lm, S_lm):
    rms_sum = 0.0
    N = len(C_lm)
    for i in range(N):
        for j in range(i + 1):
            rms_sum += (C_lm[i, j] - C_lm_true[i, j]) ** 2 + (
                S_lm[i, j] - S_lm_true[i, j]
            ) ** 2
    rms = np.sqrt(rms_sum) / (N * (N + 1))
    return rms


def main():
    max_true_deg = 30
    regress_deg = 16
    remove_deg = 0

    planet = Earth()
    sh_EGM2008 = SphericalHarmonics(planet.sh_file, regress_deg)

    test_trajectory = RandomDist(planet, [planet.radius, planet.radius + 500], 1000)
    x_test, a_test, _ = get_sh_data(
        test_trajectory,
        planet.sh_file,
        max_deg=max_true_deg,
        deg_removed=-1,
    )

    # Increase the amount of training data and see if the performance increases:
    # Performance defined as i) test data set accuracy, 2) similarity to the original harmonics

    for solver_algorithm in ["least_squares", "kaula", "single_parameter"]:
        da_vec = []
        dbeta_vec = []
        training_vec = []
        for training_size in [200, 1000, 2500, 5000]:
            train_trajectory = RandomDist(
                planet,
                [planet.radius, planet.radius + 420],
                training_size,
            )
            x, a, _ = get_sh_data(
                train_trajectory,
                planet.sh_file,
                max_deg=max_true_deg,
                deg_removed=remove_deg,
            )
            a += np.random.normal(0, 1e-3, size=a.shape)

            regressor = XuLS(regress_deg, planet, remove_deg, solver_algorithm)
            results = regressor.update(x, a)
            regress_sol = RegressSolution(results, regress_deg, remove_deg, planet)
            C_lm, S_lm = regress_sol.C_lm, regress_sol.S_lm

            k = len(C_lm)
            true_C_lm = sh_EGM2008.C_lm[:k, :k]
            true_S_lm = sh_EGM2008.S_lm[:k, :k]

            C_lm_error = (true_C_lm - C_lm) / true_C_lm * 100
            S_lm_error = (true_S_lm - S_lm) / true_S_lm * 100

            C_lm_error = np.nan_to_num(C_lm_error, posinf=0, neginf=0)
            S_lm_error = np.nan_to_num(S_lm_error, posinf=0, neginf=0)

            sh_test = SphericalHarmonics(regress_sol, regress_deg)
            a_pred = sh_test.compute_acceleration(x_test)

            da = np.average(
                np.linalg.norm(a_pred - a_test, axis=1)
                / np.linalg.norm(a_test, axis=1)
                * 100,
            )
            dbeta = compute_dbeta(true_C_lm, true_S_lm, C_lm, S_lm)

            da_vec.append(da)
            dbeta_vec.append(dbeta)
            training_vec.append(training_size)

        plt.figure(1)
        plt.semilogy(training_vec, da_vec, label=solver_algorithm)
        plt.ylabel(r"\delta a")
        plt.xlabel("Training Data Size")
        plt.legend()
        plt.figure(2)
        plt.semilogy(training_vec, dbeta_vec, label=solver_algorithm)
        plt.ylabel(r"\delta \beta")
        plt.xlabel("Training Data Size")
        plt.legend()

    plt.figure()
    analyzer = AnalyzeRegression(true_C_lm, true_S_lm, C_lm, S_lm)
    analyzer.plot_coef_rms(true_C_lm, true_S_lm)
    # analyzer.plot_coef_rms(C_lm, S_lm)
    analyzer.plot_coef_rms(C_lm - true_C_lm, S_lm - true_S_lm)
    plt.xlabel("Degree")
    plt.show()


def main():
    max_true_deg = 50
    remove_deg = 0

    planet = Earth()
    sh_EGM2008 = SphericalHarmonics(planet.sh_file, max_true_deg)

    test_trajectory = RandomDist(planet, [planet.radius, planet.radius + 500], 1000)
    x_test, a_test, _ = get_sh_data(
        test_trajectory,
        planet.sh_file,
        max_deg=max_true_deg,
        deg_removed=-1,
    )

    train_trajectory = RandomDist(planet, [planet.radius, planet.radius + 420], 5000)
    x, a, _ = get_sh_data(
        train_trajectory,
        planet.sh_file,
        max_deg=max_true_deg,
        deg_removed=remove_deg,
    )
    a += np.random.normal(0, 1e-3, size=a.shape)

    # Increase the amount of training data and see if the performance increases:
    # Performance defined as i) test data set accuracy, 2) similarity to the original harmonics

    for solver_algorithm in ["least_squares", "kaula", "kaula_inv", "single_parameter"]:
        da_vec = []
        dbeta_vec = []
        degree_vec = []
        for degree in [4, 8, 16]:
            regressor = XuLS(degree, planet, remove_deg, solver_algorithm)
            results = regressor.update(x, a)
            regress_sol = RegressSolution(results, degree, remove_deg, planet)
            C_lm, S_lm = regress_sol.C_lm, regress_sol.S_lm

            k = len(C_lm)
            true_C_lm = sh_EGM2008.C_lm[:k, :k]
            true_S_lm = sh_EGM2008.S_lm[:k, :k]

            C_lm_error = (true_C_lm - C_lm) / true_C_lm * 100
            S_lm_error = (true_S_lm - S_lm) / true_S_lm * 100

            C_lm_error = np.nan_to_num(C_lm_error, posinf=0, neginf=0)
            S_lm_error = np.nan_to_num(S_lm_error, posinf=0, neginf=0)

            sh_test = SphericalHarmonics(regress_sol, degree)
            a_pred = sh_test.compute_acceleration(x_test)

            da = np.average(
                np.linalg.norm(a_pred - a_test, axis=1)
                / np.linalg.norm(a_test, axis=1)
                * 100,
            )
            dbeta = compute_dbeta(true_C_lm, true_S_lm, C_lm, S_lm)

            da_vec.append(da)
            dbeta_vec.append(dbeta)
            degree_vec.append(degree)

        plt.figure(1)
        plt.semilogy(degree_vec, da_vec, label=solver_algorithm)
        plt.ylabel(r"\delta a")
        plt.xlabel("Degree")
        plt.legend()
        plt.figure(2)
        plt.semilogy(degree_vec, dbeta_vec, label=solver_algorithm)
        plt.ylabel(r"\delta \beta")
        plt.xlabel("Degree")
        plt.legend()

    plt.figure()
    analyzer = AnalyzeRegression(true_C_lm, true_S_lm, C_lm, S_lm)
    analyzer.plot_coef_rms(true_C_lm, true_S_lm)
    # analyzer.plot_coef_rms(C_lm, S_lm)
    analyzer.plot_coef_rms(C_lm - true_C_lm, S_lm - true_S_lm)
    plt.xlabel("Degree")
    plt.show()


if __name__ == "__main__":
    main()
