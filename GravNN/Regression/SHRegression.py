import numpy as np

from GravNN.Regression.utils import *
from GravNN.Regression.utils import compute_A, compute_euler, getK


def iterate_lstsq(M, aVec, iterations):
    results = np.linalg.lstsq(M, aVec)[0]
    delta_a = aVec - np.dot(M, results)
    for i in range(iterations):
        delta_coef = np.linalg.lstsq(M, delta_a)[0]
        results -= delta_coef
        delta_a = aVec - np.dot(M, results)
    return results


@njit(cache=True)  # , parallel=True)
def populate_H_singular(rVec1D, A, n1, n2, N, a, mu, remove_deg):
    P = len(rVec1D)
    k = remove_deg
    M = np.zeros((P, (N + 2) * (N + 1) - (k + 2) * (k + 1)))

    rVal = rVec1D[0:3]
    rMag = np.linalg.norm(rVal)
    s, t, u = rVal / rMag

    # populate variables
    A = compute_A(A, n1, n2, u)
    rE, iM, rho = compute_euler(N, a, mu, rMag, s, t)

    # NOTE: NO ESTIMATION OF C00, C10, C11 -- THESE ARE DETERMINED ALREADY
    for n in range(k + 1, N + 1):
        for m in range(0, n + 1):
            delta_m = 1 if (m == 0) else 0
            delta_m_p1 = 1 if (m + 1 == 0) else 0
            n_lm_n_lm_p1 = np.sqrt(
                (n - m) * (2.0 - delta_m) * (n + m + 1.0) / (2.0 - delta_m_p1),
            )
            n_lm_n_l_p1_m_p1 = np.sqrt(
                (n + m + 2.0)
                * (n + m + 1.0)
                * (2.0 * n + 1.0)
                * (2.0 - delta_m)
                / ((2.0 * n + 3.0) * (2.0 - delta_m_p1)),
            )

            c1 = n_lm_n_lm_p1  # Eq 79 BSK
            c2 = n_lm_n_l_p1_m_p1  # Eq 80 BSK

            # TODO: These will need the normalizaiton factor out in front (N1, N2)
            # Coefficient contribution to X, Y, Z components of the acceleration

            if m == 0:
                rTerm = 0
                iTerm = 0
            else:
                rTerm = rE[m - 1]
                iTerm = iM[m - 1]

            # Pines Derivatives -- but rho n+1 rather than n+2
            f_Cnm_1 = (rho[n + 1] / a) * (
                m * A[n, m] * rTerm - s * c2 * A[n + 1, m + 1] * rE[m]
            )
            f_Cnm_2 = (rho[n + 1] / a) * (
                -m * A[n, m] * iTerm - t * c2 * A[n + 1, m + 1] * rE[m]
            )

            if m < n:
                f_Cnm_3 = (
                    (rho[n + 1] / a)
                    * (c1 * A[n, m + 1] - u * c2 * A[n + 1, m + 1])
                    * rE[m]
                )
            else:
                f_Cnm_3 = (rho[n + 1] / a) * (-1.0 * u * c2 * A[n + 1, m + 1]) * rE[m]

            f_Snm_1 = (rho[n + 1] / a) * (
                m * A[n, m] * iTerm - s * c2 * A[n + 1, m + 1] * iM[m]
            )
            f_Snm_2 = (rho[n + 1] / a) * (
                m * A[n, m] * rTerm - t * c2 * A[n + 1, m + 1] * iM[m]
            )
            if m < n:
                f_Snm_3 = (
                    (rho[n + 1] / a)
                    * (c1 * A[n, m + 1] - u * c2 * A[n + 1, m + 1])
                    * iM[m]
                )
            else:
                f_Snm_3 = (rho[n + 1] / a) * (-1.0 * u * c2 * A[n + 1, m + 1]) * iM[m]

            degIdx = (n + 1) * (n) - (k + 2) * (k + 1)

            M[0, degIdx + 2 * m + 0] = f_Cnm_1  # X direction
            M[0, degIdx + 2 * m + 1] = f_Snm_1
            M[1, degIdx + 2 * m + 0] = f_Cnm_2  # Y direction
            M[1, degIdx + 2 * m + 1] = f_Snm_2
            M[2, degIdx + 2 * m + 0] = f_Cnm_3  # Z direction
            M[2, degIdx + 2 * m + 1] = f_Snm_3

    return M


@njit(cache=True)  # , parallel=True)
def populate_M(rVec1D, A, n1, n2, N, a, mu, remove_deg):
    P = len(rVec1D)
    k = remove_deg
    M = np.zeros((P, (N + 2) * (N + 1) - (k + 2) * (k + 1)))

    for p in range(0, int(P / 3)):
        rVal = rVec1D[3 * p : 3 * (p + 1)]
        H = populate_H_singular(rVal, A, n1, n2, N, a, mu, remove_deg)
        M[3 * p + 0 : 3 * p + 3, :] = H

    return M


class SHRegression:
    def __init__(self, N, a, mu, M):
        self.N = N
        self.a = a
        self.mu = mu
        self.M = M

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

    def populate_M(self, rVec1D, remove_deg):
        return populate_M(
            rVec1D,
            self.A,
            self.n1,
            self.n2,
            self.N,
            self.a,
            self.mu,
            remove_deg,
        )

    def populate_H_singular(self, rVec1D, remove_deg):
        return populate_H_singular(
            rVec1D,
            self.A,
            self.n1,
            self.n2,
            self.N,
            self.a,
            self.mu,
            remove_deg,
        )

    def batch(self, rVec, aVec, iterations=5):
        self.rVec1D = rVec.reshape((-1,))
        self.aVec1D = aVec.reshape((-1,))
        self.P = len(self.rVec1D)

        M = self.SHRegressor.populate_M(
            self.rVec1D,
            self.remove_deg,
        )
        results = iterate_lstsq(M, self.aVec1D, iterations)
        return results

    def recursive(self, x, y, init_batch=None):
        if init_batch is not None:
            self.batch(x[:init_batch, :], y[:init_batch, :])
        else:
            np.zeros((self.N + 2) * (self.N + 1))

        # Reminder: x is the vector of coefficients
        # y is the acceleration measured
        # Hk is the partial of da/dCoef(r) | r=r_k
        xk_m_1 = self.x_hat
        Pk_m_1 = self.P_hat
        Rk = self.Rk

        Hk = self.SHRegressor.populate_H_singular(
            rk,
            self.remove_deg,
        )
        sub_K_inv = np.linalg.inv(Rk + np.dot(Hk, np.dot(Pk_m_1, Hk.T)))
        Kk = np.dot(Pk_m_1, np.dot(Hk.T, sub_K_inv))

        Pk_sub = np.identity(len(xk_m_1)) - np.dot(Kk, Hk)
        Pk = np.dot(Pk_sub, np.dot(Pk_m_1, Pk_sub.T))

        self.x_hat = xk_m_1 + np.dot(Kk, yk - np.dot(Hk, xk_m_1))
        self.P_hat = Pk

        return
