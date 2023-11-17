import numpy as np

from GravNN.Regression.utils import *
from GravNN.Regression.utils import compute_A, compute_euler, getK
from GravNN.Support.ProgressBar import ProgressBar


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
    def __init__(
        self,
        max_degree,
        min_degree,
        planet_radius,
        planet_mu,
        kaula_factor=0.0,
        max_batch_size=-1,
    ):
        self.N = max_degree
        self.M = min_degree
        self.a = planet_radius
        self.mu = planet_mu
        self.kaula_factor = kaula_factor
        self.max_batch_size = max_batch_size

        self.rE = np.zeros((self.N + 2,))
        self.iM = np.zeros((self.N + 2,))
        self.rho = np.zeros((self.N + 3,))

        self.A = np.zeros((self.N + 2, self.N + 2))

        self.n1 = np.zeros((self.N + 2, self.N + 2))
        self.n2 = np.zeros((self.N + 2, self.N + 2))
        self.init_calculations()
        self.count_total_coefficients()
        self.compute_kaula_matrix()

    def count_total_coefficients(self):
        self.terms_total = int((self.N + 1) * (self.N + 2))
        self.terms_removed = int((self.M + 1) * (self.M + 2))
        self.terms_remaining = self.terms_total - self.terms_removed

    def init_calculations(self):
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

    def compute_kaula_matrix(self):
        l = self.M + 1
        m = 0

        # Generate the full matrix up to the max degree
        diag = np.ones((self.terms_remaining))
        kaula = np.diag(diag)

        for i in range(self.terms_removed, self.terms_total):
            if l != 0:
                kaula[i, i] = (1 / l**2) ** -1

            # every odd number, increment the m index
            # because we've iterated over a C and S pair
            if (i + 1) % 2 == 0:
                if l == m:  #
                    l += 1
                    m = 0
                else:
                    m += 1
        self.kaula = kaula

    def populate_M(self, rVec1D):
        return populate_M(
            rVec1D,
            self.A,
            self.n1,
            self.n2,
            self.N,
            self.a,
            self.mu,
            self.M,
        )

    def batch(self, rVec, aVec):
        rVec1D = rVec.reshape((-1,))
        aVec1D = aVec.reshape((-1,))
        self.P = len(rVec1D)

        M = self.populate_M(rVec1D)

        inv_arg = M.T @ M

        # Compute the Least Squares Solution
        ridge = self.kaula_factor * self.kaula
        inv_arg = M.T @ M
        ridge = self.kaula_factor * self.kaula
        K_inv_k = np.linalg.inv(inv_arg + ridge)

        self.x_hat = K_inv_k @ M.T @ aVec1D
        self.K_inv_k = K_inv_k

        return self.x_hat

    def recursive_batch(self, rk, yk):
        # Load current estimates
        xk = self.x_hat

        # Populate partials
        Hk = self.populate_M(rk)
        I = np.identity(len(Hk))
        inter_inv = np.linalg.inv(I + Hk @ self.K_inv_k @ Hk.T)
        K_inv_kp1 = self.K_inv_k - self.K_inv_k @ Hk.T @ inter_inv @ Hk @ self.K_inv_k
        xk_p1 = xk + K_inv_kp1 @ Hk.T @ (yk - Hk @ xk)

        # update estimates
        self.x_hat = xk_p1
        self.K_inv_k = K_inv_kp1
        return

    def recursive(self, r, y):
        BS = self.max_batch_size
        r_init = r[:BS]
        y_init = y[:BS]

        # Need first guess before you can begin
        # recursive
        self.batch(r_init, y_init)

        pbar = ProgressBar(len(r), enable=True)
        for i in range(BS, len(r), BS):
            end_idx = min(i + BS, len(r))
            rBatch = r[i:end_idx].reshape((-1,))
            yBatch = y[i:end_idx].reshape((-1,))
            self.recursive_batch(rBatch, yBatch)
            pbar.update(end_idx)

        return self.x_hat

    def update(self, rVec, aVec):
        unlimited_batch = True if self.max_batch_size == -1 else False
        small_batch = True if len(rVec) < self.max_batch_size else False

        if unlimited_batch or small_batch:
            results = self.batch(rVec, aVec)
        else:
            results = self.recursive(rVec, aVec)

        return results
