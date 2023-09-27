import numpy as np

from GravNN.Regression.utils import (
    format_coefficients,
    getK,
    populate_H_singular,
    populate_removed_degrees,
    save,
)


class RLLS:
    def __init__(self, max_deg, planet, x0, P0, Rk, remove_deg=-1):
        self.N = max_deg  # Degree
        self.a = planet.radius
        self.mu = planet.mu

        self.x_hat = x0.reshape((-1,))
        self.P_hat = P0
        self.Rk = Rk
        self.remove_deg = remove_deg

        self.rE = np.zeros((self.N + 2,))
        self.iM = np.zeros((self.N + 2,))
        self.rho = np.zeros((self.N + 3,))

        self.A = np.zeros((self.N + 2, self.N + 2))

        self.n1 = np.zeros((self.N + 2, self.N + 2))
        self.n2 = np.zeros((self.N + 2, self.N + 2))

        for i in range(0, self.N + 2):
            if i == 0:
                self.A[i, i] = 1.0
            else:
                self.A[i, i] = (
                    np.sqrt((2.0 * i + 1.0) * getK(i) / (2.0 * i * getK(i - 1)))
                    * self.A[i - 1, i - 1]
                )

            for m in range(0, i + 1):  # Check the plus one
                if i >= m + 2:
                    self.n1[i, m] = np.sqrt(
                        ((2.0 * i + 1.0) * (2.0 * i - 1.0)) / ((i - m) * (i + m)),
                    )
                    self.n2[i, m] = np.sqrt(
                        ((i + m - 1.0) * (2.0 * i + 1.0) * (i - m - 1.0))
                        / ((i + m) * (i - m) * (2.0 * i - 3.0)),
                    )

    def update(self, rk, yk):
        # Reminder: x is the vector of coefficients
        # y is the acceleration measured
        # Hk is the partial of da/dCoef(r) | r=r_k
        xk_m_1 = self.x_hat
        Pk_m_1 = self.P_hat
        Rk = self.Rk

        Hk = populate_H_singular(
            rk,
            self.A,
            self.n1,
            self.n2,
            self.N,
            self.a,
            self.mu,
            self.remove_deg,
        )
        sub_K_inv = np.linalg.inv(Rk + np.dot(Hk, np.dot(Pk_m_1, Hk.T)))
        Kk = np.dot(Pk_m_1, np.dot(Hk.T, sub_K_inv))

        Pk_sub = np.identity(len(xk_m_1)) - np.dot(Kk, Hk)
        Pk = np.dot(Pk_sub, np.dot(Pk_m_1, Pk_sub.T))

        self.x_hat = xk_m_1 + np.dot(Kk, yk - np.dot(Hk, xk_m_1))
        self.P_hat = Pk

        return


def main():
    import matplotlib.pyplot as plt

    from GravNN.CelestialBodies.Planets import Earth
    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
    from GravNN.Regression.BLLS import BLLS
    from GravNN.Support.ProgressBar import ProgressBar
    from GravNN.Trajectories import DHGridDist, RandomDist

    planet = Earth()

    max_true_deg = 10
    regress_deg = 2
    remove_deg = -1
    batch_initialization = True
    sh_EGM2008 = SphericalHarmonics(planet.sh_file, regress_deg)

    trajectory = DHGridDist(planet, sh_EGM2008.radEquator, 360)
    trajectory = RandomDist(planet, [planet.radius, planet.radius + 420000], 100000)

    if remove_deg != -1:
        x, a, u = get_sh_data(
            trajectory,
            planet.sh_file,
            max_deg=max_true_deg,
            deg_removed=remove_deg,
        )
        C_lm_start = 0.0
    else:
        x, a, u = get_sh_data(
            trajectory,
            planet.sh_file,
            max_deg=max_true_deg,
            deg_removed=-1,
        )
        C_lm_start = 1.0

    N = regress_deg
    M = remove_deg

    # Initial Coefficient Estimates and Covariances
    # N+1 accounts for the C00 terms
    if batch_initialization:
        batch_regressor = BLLS(N, planet, M)
        results = batch_regressor.update(x[:100, :], a[:100, :])
        x0 = results
    else:
        x0 = np.zeros(((N + 2) * (N + 1) - (M + 2) * (M + 1),))
        x0[0] = C_lm_start

    P0 = np.identity((N + 2) * (N + 1) - (M + 2) * (M + 1)) * 1e-6
    P0[np.isnan(P0)] = 0.0
    Rk = np.identity(3) * np.std(a) * 10

    # Initialize the regressor
    regressor = RLLS(N, planet, x0, P0, Rk, M)

    # Record time history of regressor
    x_hat_hist = []
    P_hat_hist = []

    pbar = ProgressBar(len(x), enable=True)
    i = 0
    for x_inst, y_inst in zip(x, a):
        regressor.update(x_inst, y_inst)
        i += 1
        pbar.update(i)

        x_hat_hist.append(regressor.x_hat)
        P_hat_hist.append(np.diag(regressor.P_hat).tolist())

    file_name = "/Users/johnmartin/Documents/GraduateSchool/Research/ML_Gravity/GravNN/Files/GravityModels/Regressed/Earth/test.csv"
    save(file_name, planet, regressor.x_hat, regress_deg, remove_deg)

    C_lm, S_lm = format_coefficients(regressor.x_hat, regressor.N, regressor.remove_deg)
    C_lm, S_lm = populate_removed_degrees(
        C_lm,
        S_lm,
        sh_EGM2008.C_lm,
        sh_EGM2008.S_lm,
        remove_deg,
    )

    plot_coef_history(x_hat_hist, P_hat_hist, sh_EGM2008, remove_deg, start_idx=0)

    plt.show()


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


if __name__ == "__main__":
    main()
