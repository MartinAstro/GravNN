import tempfile

import numpy as np

from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.SHRegression import SHRegression
from GravNN.Regression.utils import (
    format_coefficients,
    populate_removed_degrees,
    save,
)
from GravNN.Support.ProgressBar import ProgressBar


class RLLS:
    def __init__(self, max_deg, planet, x0, P0, Rk, remove_deg=-1):
        self.N = max_deg  # Degree
        self.planet = planet
        self.remove_deg = remove_deg
        self.x_hat = x0.reshape((-1,))
        self.P_hat = P0
        self.Rk = Rk
        self.SHRegressor = SHRegression(max_deg, remove_deg, planet.radius, planet.mu)
        self.initialized = False
        self.x0 = np.zeros((len(x0),))

    def batch_start(self, x, a):
        init_degree = self.N if self.N < 10 else 10
        batch_regressor = BLLS(init_degree, self.planet, self.remove_deg)
        init_results = batch_regressor.update(x, a)
        results_dim = len(init_results)
        self.x_hat[:results_dim] = init_results

    def update_single(self, rk, yk):
        # Load current estimates
        xk_m_1 = self.x_hat
        Pk_m_1 = self.P_hat
        Rk = self.Rk

        # Populate partials
        Hk = self.SHRegressor.populate_H_singular(
            rk,
            self.remove_deg,
        )
        sub_K_inv = np.linalg.inv(Rk + np.dot(Hk, np.dot(Pk_m_1, Hk.T)))
        Kk = np.dot(Pk_m_1, np.dot(Hk.T, sub_K_inv))

        Pk_sub = np.identity(len(xk_m_1)) - np.dot(Kk, Hk)

        assert np.all(np.isfinite(Pk_m_1))
        assert np.all(np.isfinite(Hk))
        assert np.all(np.isfinite(sub_K_inv))
        assert np.all(np.isfinite(Kk))
        assert np.all(np.isfinite(Pk_sub))

        # update estimates
        self.x_hat = xk_m_1 + np.dot(Kk, yk - np.dot(Hk, xk_m_1))
        self.P_hat = np.dot(Pk_sub, np.dot(Pk_m_1, Pk_sub.T))
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
        for i in range(init_batch, len(r)):
            self.update_single(r[i], y[i])
            pbar.update(i)

            # optionally save
            if history:
                self.x_hat_hist.append(self.x_hat)
                self.P_hat_hist.append(np.diag(self.P_hat).tolist())

        return self.x_hat


class RLLS2:
    def __init__(self, max_deg, planet, x0, alpha=1e-8, remove_deg=-1):
        self.N = max_deg  # Degree
        self.planet = planet
        self.remove_deg = remove_deg
        self.x_hat = x0.reshape((-1,))
        self.SHRegressor = SHRegression(max_deg, remove_deg, planet.radius, planet.mu)
        self.initialized = False
        self.alpha = alpha
        self.x0 = np.zeros((len(x0),))

    def batch_start(self, x, a):
        init_degree = self.N if self.N < 10 else 10
        batch_regressor = BLLS(init_degree, self.planet, self.remove_deg)
        init_results = batch_regressor.update(x, a)
        results_dim = len(init_results)
        self.x_hat[:results_dim] = init_results

        H = self.SHRegressor.populate_M(x, self.remove_deg)
        I = np.identity(len(self.x_hat))
        self.K_inv_k = np.linalg.inv(H.T @ H + self.alpha * I)

    def update_single(self, rk, yk):
        # Load current estimates
        xk = self.x_hat

        # Populate partials
        Hk = self.SHRegressor.populate_H_singular(
            rk,
            self.remove_deg,
        )

        I = np.identity(len(self.x_hat))
        inter_inv = np.linalg.inv(I + Hk @ self.K_inv_k @ Hk.T)
        K_inv_kp1 = self.K_inv_k - self.K_inv_k @ Hk.T @ inter_inv @ Hk @ self.K_inv_k
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
        for i in range(init_batch, len(r)):
            self.update_single(r[i], y[i])
            pbar.update(i)

            # optionally save
            if history:
                self.x_hat_hist.append(self.x_hat)
                self.P_hat_hist.append(np.diag(self.P_hat).tolist())

        return self.x_hat


def plot_coef_history(x_hat_hist, P_hat_hist, sh_EGM2008, remove_deg, start_idx=0):
    import matplotlib.pyplot as plt

    x_hat_hist = np.array(x_hat_hist)
    P_hat_hist = np.array(P_hat_hist)

    l = remove_deg + 1
    m = 0

    for i in range(len(x_hat_hist[0])):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(x_hat_hist[start_idx:, i], c="b")
        plt.plot(
            x_hat_hist[start_idx:, i] + 3 * np.sqrt(P_hat_hist[start_idx:, i]),
            c="r",
        )
        plt.plot(
            x_hat_hist[start_idx:, i] - 3 * np.sqrt(P_hat_hist[start_idx:, i]),
            c="r",
        )

        C_lm = sh_EGM2008.C_lm[l, m]
        S_lm = sh_EGM2008.S_lm[l, m]

        if i % 2 == 0:
            coef = C_lm
            plt.suptitle("C" + str(l) + str(m))
        else:
            coef = S_lm
            plt.suptitle("S" + str(l) + str(m))

        plt.subplot(2, 1, 2)
        plt.plot(x_hat_hist[start_idx:, i] - coef, c="b")
        plt.plot(
            x_hat_hist[start_idx:, i] - coef + 3 * np.sqrt(P_hat_hist[start_idx:, i]),
            c="r",
        )
        plt.plot(
            x_hat_hist[start_idx:, i] - coef - 3 * np.sqrt(P_hat_hist[start_idx:, i]),
            c="r",
        )

        if i % 2 != 0:
            if m < l:
                m += 1
            else:
                l += 1
                m = 0


def test_setup(max_true_degree, regress_degree, remove_degree, P0_diag, initial_batch):
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

    # Define uncertainty
    P0 = np.identity(dim) * P0_diag

    # Functionally no meas uncertainty, but 1e-16 needed to avoid singularities
    Rk = np.identity(3) + 1e-16

    # Initialize the regressor
    regressor = RLLS(REGRESS_DEG, planet, x0, P0, Rk, REMOVE_DEG)
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
    # # Slightly wrong
    a_error = test_setup(
        max_true_degree=10,
        regress_degree=4,
        remove_degree=1,
        P0_diag=0,  # assume only uncertainty in measurements (not in initial guess)
        initial_batch=1000,
    )
    assert np.isclose(a_error, 0.010926797140351335)

    # Perfect
    a_error = test_setup(
        max_true_degree=4,
        regress_degree=4,
        remove_degree=-1,
        P0_diag=0,
        initial_batch=500,
    )
    assert np.isclose(a_error, 2.705730413079536e-11)


if __name__ == "__main__":
    main()
