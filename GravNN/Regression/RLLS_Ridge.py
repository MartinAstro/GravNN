import tempfile

import numpy as np
from numba import njit

from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.SHRegression import SHRegression
from GravNN.Regression.utils import (
    format_coefficients,
    populate_removed_degrees,
    save,
)
from GravNN.Support.ProgressBar import ProgressBar


@njit(cache=True)  # , parallel=True)
def update_K(K_inv_k, Hk, N):
    I = np.identity(N)
    inter_inv = np.linalg.inv(I + Hk @ K_inv_k @ Hk.T)
    K_inv_kp1 = K_inv_k - K_inv_k @ Hk.T @ inter_inv @ Hk @ K_inv_k
    return K_inv_kp1


class RLLS_Ridge:
    def __init__(self, max_deg, planet, x0, alpha=1e-8, remove_deg=-1, batch_size=1):
        self.N = max_deg  # Degree
        self.planet = planet
        self.remove_deg = remove_deg
        self.x_hat = x0.reshape((-1,))
        self.SHRegressor = SHRegression(max_deg, planet.radius, planet.mu, remove_deg)
        self.initialized = False
        self.alpha = alpha
        self.x0 = np.zeros((len(x0),))
        self.batch_size = batch_size

    def batch_start(self, x, a):
        init_degree = self.N  # if self.N < 10 else 10
        batch_regressor = BLLS(
            init_degree,
            self.planet,
            self.remove_deg,
            ridge_factor=self.alpha,
        )

        # Run a batch update
        init_coef = batch_regressor.update(x, a)
        dim = len(init_coef)
        self.x_hat[:dim] = init_coef

        # compute K_inv
        x1D = x.reshape((-1,))
        H = self.SHRegressor.populate_M(x1D, self.remove_deg)
        inv_arg = H.T @ H
        batch_regressor.SHRegressor.K_inv = self.SHRegressor.K_inv
        ridge = batch_regressor.compute_ridge(inv_arg)
        self.K_inv_k = np.linalg.inv(inv_arg + ridge)

    def update_batch(self, rk, yk):
        # Load current estimates
        xk = self.x_hat

        # Populate partials
        Hk = self.SHRegressor.populate_M(
            rk,
            self.remove_deg,
        )

        # Compute estimates
        K_inv_kp1 = update_K(self.K_inv_k, Hk, len(rk))
        xk_p1 = xk + K_inv_kp1 @ Hk.T @ (yk - Hk @ xk)

        # update estimates
        self.x_hat = xk_p1
        self.K_inv_k = K_inv_kp1
        return

    def update(self, r, y, init_batch=0, history=False):
        # Record time history of regressor
        self.x_hat_hist = []
        self.P_hat_hist = []

        # Populate an initial guess
        if init_batch > 0:
            r_start = r[:init_batch, :]
            y_start = y[:init_batch, :]
            self.batch_start(r_start, y_start)

        # Update based on incoming data
        pbar = ProgressBar(len(r), enable=True)
        BS = self.batch_size
        for i in range(init_batch, len(r), BS):
            end_idx = min(i + BS, len(r))
            rBatch = r[i:end_idx].reshape((-1,))
            yBatch = y[i:end_idx].reshape((-1,))
            self.update_batch(rBatch, yBatch)
            pbar.update(end_idx)

            # optionally save
            if history:
                self.x_hat_hist.append(self.x_hat)

        return self.x_hat


def test_setup(max_true_degree, regress_degree, remove_degree, initial_batch):
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
    regressor = RLLS_Ridge(REGRESS_DEG, planet, x0, alpha=1e-3, remove_deg=REMOVE_DEG)
    regressor.update(x, a, init_batch=initial_batch, history=True)

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
        print(f"\n ACCELERATION ERROR: {np.mean(a_error)}")

    return np.mean(a_error)


def main():
    # # Slightly wrong
    a_error = test_setup(
        max_true_degree=100,
        regress_degree=90,
        remove_degree=-1,
        initial_batch=1000,
    )
    assert np.isclose(a_error, 0.0015995795115804907)

    # a_error = test_setup(
    #     max_true_degree=10,
    #     regress_degree=4,
    #     remove_degree=1,
    #     initial_batch=1000,
    # )
    # assert np.isclose(a_error, 0.0015995795115804907)

    # # Perfect
    # a_error = test_setup(
    #     max_true_degree=4,
    #     regress_degree=4,
    #     remove_degree=-1,
    #     initial_batch=500,
    # )
    # assert np.isclose(a_error, 1.0390206620479445e-12)


if __name__ == "__main__":
    main()
