import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import sigfig
from numba import njit
from scipy.optimize import Bounds, minimize

from GravNN.Regression.utils import compute_A, compute_euler, format_coefficients, getK


@njit(cache=True)  # , parallel=True)
def populate_M(rVec1D, A, n1, n2, N, a, mu, remove_deg):
    P = len(rVec1D)
    k = remove_deg
    M = np.zeros((P, (N + 2) * (N + 1) - (k + 2) * (k + 1)))

    for p in range(0, int(P / 3)):
        rVal = rVec1D[3 * p : 3 * (p + 1)]
        rMag = np.linalg.norm(rVal)
        s, t, u = rVal / rMag

        # populate variables
        A = compute_A(A, n1, n2, u)
        rE, iM, rho = compute_euler(N, a, mu, rMag, s, t)

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

                # TODO: These will need the normalization factor out in front (N1, N2)
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
                    f_Cnm_3 = (
                        (rho[n + 1] / a) * (-1.0 * u * c2 * A[n + 1, m + 1]) * rE[m]
                    )

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
                    f_Snm_3 = (
                        (rho[n + 1] / a) * (-1.0 * u * c2 * A[n + 1, m + 1]) * iM[m]
                    )

                degIdx = (n + 1) * (n) - (k + 2) * (k + 1)

                M[3 * p + 0, degIdx + 2 * m + 0] = f_Cnm_1  # X direction
                M[3 * p + 0, degIdx + 2 * m + 1] = f_Snm_1
                M[3 * p + 1, degIdx + 2 * m + 0] = f_Cnm_2  # Y direction
                M[3 * p + 1, degIdx + 2 * m + 1] = f_Snm_2
                M[3 * p + 2, degIdx + 2 * m + 0] = f_Cnm_3  # Z direction
                M[3 * p + 2, degIdx + 2 * m + 1] = f_Snm_3

    return M


class XuLS:
    def __init__(self, max_deg, planet, remove_deg=-1, algorithm="kaula"):
        self.algorithm = algorithm
        self.N = max_deg  # Degree
        self.a = planet.radius
        self.mu = planet.mu
        self.remove_deg = remove_deg
        self.compute_kaula_matrix()

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

    def compute_kaula_matrix(self):
        l = self.remove_deg + 1
        m = 0
        factor = 1
        q = self.remove_deg
        terms = int((self.N + 1) * (self.N + 2))
        terms_removed = int((q + 1) * (q + 2))
        K = np.diag(np.zeros((terms - terms_removed)))
        K_inv = np.diag(np.zeros((terms - terms_removed)))
        for i in range(0, len(K)):  # all coefficients (C and S) excluding C_00, S_00
            K[i, i] = (factor / l**2) ** 1
            K_inv[i, i] = (factor / l**2) ** -1
            if (
                i + 1
            ) % 2 == 0:  # every odd number, increment the m index (because we've iterated over a C and S pair)
                if l == m:  #
                    l += 1
                    m = 0
                else:
                    m += 1
        self.K = K
        self.K_inv = K_inv

    def compute_coefficients(self, A, Y, R):
        if self.algorithm == "least_squares":
            q = len(A[0])
            K = np.zeros((q, q))
            coef = self.ridge_regression(A, Y, R, K)
        if self.algorithm == "kaula":
            q = len(A[0])
            K = self.K_inv / 1e7
            coef = self.ridge_regression(A, Y, R, K)

        if self.algorithm == "single_parameter":
            # coef = self.single_parameter_regression(A, Y, R)
            coef = self.modified_single_parameter_regression(A, Y, R)
        if self.algorithm == "single_parameter_kaula":
            # coef = self.single_parameter_kaula_regression(A, Y, R)
            coef = self.modified_single_parameter_kaula_regression(A, Y, R)
        if self.algorithm == "xu_rummel_94":
            coef = self.generalized_ridge_regression(A, Y, R)
            # coef = generalized_ridge_regression(A, Y, R)
        if self.algorithm == "custom":
            coef = self.custom_single_parameter_regression(A, Y, R)
        return coef

    def ridge_regression(self, A, Y, R, K):
        # K matrix can be interpreted as prior information about the state (Bayesian)
        # or simply a means through which the matrix is better conditioned and biased (frequentist)
        # if K = 0 -> weighted least squares
        # if R = identity -> unweighted least squares

        R_diag = R.diagonal()
        P = sp.sparse.diags(1.0 / R_diag)
        X = np.linalg.pinv(A.T @ P @ A + K) @ A.T @ P @ Y
        return X

    def generalized_ridge_regression(self, A, Y, R):
        def check_symmetric(a, rtol=1e-05, atol=1e-08):
            return np.allclose(a, a.T, rtol=rtol, atol=atol)

        R_diag = R.diagonal()
        sigma = np.sqrt(np.min(R_diag))  # highest weights are those with smallest error
        # P_inv = sp.sparse.diags(R_diag/sigma**2)
        P = sp.sparse.diags(sigma**2 / R_diag)

        # because A.T@A is a positive-semidefinite matrix, SVD and eigendecomp are
        # synonymous, but SVD is faster to  compute.
        u, svd_eig_val, v = np.linalg.svd(A.T @ P @ A, hermitian=True)

        eig_vec = u
        eig_val = svd_eig_val

        # u and svd eig_val are just the flipped entries of np_eig_val
        Lambda = np.diag(eig_val)
        G = eig_vec
        A1 = A @ G

        # without ridge regression
        alpha_hat = np.linalg.pinv(Lambda) @ A1.T @ P @ Y
        alpha_g_hat = np.zeros_like(alpha_hat)

        K = np.array([sigma**2 / alpha_hat[i] ** 2 for i in range(len(alpha_hat))])
        K[K == np.inf] = 0
        alpha_g_hat = np.linalg.pinv(Lambda + K) @ A1.T @ P @ Y

        X_g = G @ alpha_g_hat
        return X_g

    def modified_generalized_ridge_regression(self, A, Y, R):
        """an attempt to include sigma_i instead of constant sigma"""

        def check_symmetric(a, rtol=1e-05, atol=1e-08):
            return np.allclose(a, a.T, rtol=rtol, atol=atol)

        sigma = np.sqrt(R.diagonal())

        # because A.T@A is a positive-semidefinite matrix, SVD and eigendecomp are
        # synonymous, but SVD is faster to  compute.
        u, svd_eig_val, v = np.linalg.svd(A.T @ R @ A, hermitian=True)

        eig_vec = u
        eig_val = svd_eig_val

        # u and svd eig_val are just the flipped entries of np_eig_val
        Lambda = np.diag(eig_val)
        G = eig_vec
        A1 = A @ G

        # without ridge regression
        alpha_hat = np.linalg.pinv(Lambda) @ A1.T @ R @ Y
        alpha_g_hat = np.zeros_like(alpha_hat)

        # Iterate over k
        iterations = 0
        while np.linalg.norm(alpha_g_hat - alpha_hat) > 1e-12 and iterations < 5:
            if iterations != 0:
                alpha_hat = alpha_g_hat
            # with ridge regression
            K = np.array(
                [sigma[i] ** 2 / alpha_hat[i] ** 2 for i in range(len(alpha_hat))],
            )
            K[K == np.inf] = 0
            alpha_g_hat = np.linalg.pinv(Lambda + K) @ A1.T @ R @ Y
            iterations += 1
            print(
                f"Iteration {iterations} \t dk={np.linalg.norm(alpha_g_hat - alpha_hat)}",
            )

        X_g = G @ alpha_g_hat
        return X_g

    def single_parameter_kaula_regression(self, A, Y, R):
        np.sqrt(R.diagonal())
        k = 1e-5  # to be optimized for
        Q = len(A[0])
        K0 = self.K_inv[:Q, :Q]
        # l_max = 2
        # l_max_idx = (l_max+1)*(l_max+2) - 1 #
        # K0[:l_max_idx, :l_max_idx] = 0.0 # don't impose kaula's rule for l < 4

        # Remove the zero rows from the invertible matrix
        ARA = A.T @ R @ A
        # empty_rows = np.array([np.all(row == 0.0) for row in ARA])
        # for i in range(len(empty_rows)):
        #     ARA[i,i] = 1.0 if empty_rows[i] else ARA[i,i]

        # compute solution, and iterate until optimal k
        beta = np.linalg.pinv(ARA + k * K0) @ A.T @ R @ Y

        def objective(k):
            N_s = ARA + k * K0
            N_s_inv = np.linalg.pinv(N_s)
            D_beta = N_s_inv @ ARA @ N_s_inv  # R contains sigma**2
            f_s = (
                np.trace(D_beta) + k**2 * beta.T @ K0 @ N_s_inv @ N_s_inv @ K0 @ beta
            )
            return f_s

        res = minimize(objective, [k], tol=1e-16)
        print(f"Total iterations: {res.nit} \t k={res.x[0]}")
        k = res.x[0]
        return np.linalg.pinv(ARA + k * K0) @ A.T @ R @ Y

    def modified_single_parameter_kaula_regression(self, A, Y, R):
        R_diag = R.diagonal()
        P = sp.sparse.diags(1.0 / R_diag)
        # k = 1E8
        k = 1
        I0 = self.K_inv  # Multiple Parameter near Eq. (2)

        # Remove the zero rows from the invertible matrix
        APA = A.T @ P @ A

        # compute solution, and iterate until optimal k
        def objective(k):
            N_s = APA + k * I0
            N_s_inv = np.linalg.pinv(N_s)
            beta = N_s_inv @ A.T @ P @ Y
            D_beta = N_s_inv @ APA @ N_s_inv  # R contains sigma**2
            f_s = (
                np.trace(D_beta) + k**2 * beta.T @ I0 @ N_s_inv @ N_s_inv @ I0 @ beta
            )
            return f_s

        res = minimize(objective, [k], tol=1e-16)
        print(f"Total iterations: {res.nit} \t k = {res.x[0]}")
        k = res.x[0]
        return np.linalg.pinv(APA + k * I0) @ A.T @ P @ Y

    def single_parameter_regression(self, A, Y, R):
        R_diag = R.diagonal()
        sigma = np.sqrt(np.min(R_diag))  # highest weights are those with smallest error
        # P_inv = sp.sparse.diags(R_diag/sigma**2)
        P = sp.sparse.diags(sigma**2 / R_diag)

        k = 1e-5
        I0 = np.eye(len(A[0]))
        l_max = 4
        l_max_idx = (l_max + 1) * (l_max + 2)
        I0[:l_max_idx, :l_max_idx] = 0.0

        # Remove the zero rows from the invertible matrix
        APA = A.T @ P @ A
        empty_rows = np.array([np.all(row == 0.0) for row in APA])
        for i in range(len(empty_rows)):
            APA[i, i] = 1.0 if empty_rows[i] else APA[i, i]

        # compute solution, and iterate until optimal k
        beta = np.linalg.inv(APA + k * I0) @ A.T @ P @ Y

        def objective(k):
            N_s = APA + k * I0
            N_s_inv = np.linalg.inv(N_s)
            D_beta = sigma**2 * N_s_inv @ APA @ N_s_inv
            f_s = (
                np.trace(D_beta) + k**2 * beta.T @ I0 @ N_s_inv @ N_s_inv @ I0 @ beta
            )
            return f_s

        res = minimize(objective, [k], tol=1e-10)
        print(f"Total iterations: {res.nit}")
        k = res.x[0]
        return np.linalg.inv(APA + k * I0) @ A.T @ P @ Y

    def modified_single_parameter_regression(self, A, Y, R):
        R_diag = R.diagonal()
        P = sp.sparse.diags(1.0 / R_diag)
        k = 1
        I0 = np.eye(len(A[0]))

        # Remove the zero rows from the invertible matrix
        APA = A.T @ P @ A

        # compute solution, and iterate until optimal k
        def objective(k):
            N_s = APA + k * I0
            N_s_inv = np.linalg.pinv(N_s)
            beta = N_s_inv @ A.T @ P @ Y
            D_beta = N_s_inv @ APA @ N_s_inv  # R contains sigma**2
            f_s = (
                np.trace(D_beta) + k**2 * beta.T @ I0 @ N_s_inv @ N_s_inv @ I0 @ beta
            )
            return f_s

        res = minimize(objective, [k], tol=1e-16)
        print(f"Total iterations: {res.nit} \t k = {res.x[0]}")
        k = res.x[0]
        return np.linalg.pinv(APA + k * I0) @ A.T @ P @ Y

    def custom_single_parameter_regression(self, A, Y, R):
        k = 1e-5  # to be optimized for
        R_diag = R.diagonal()
        P = sp.sparse.diags(1.0 / R_diag)
        np.eye(len(A[0]))

        # Remove the zero rows from the invertible matrix
        APA = A.T @ P @ A

        Q = len(A[0])
        K0 = self.K_inv[:Q, :Q]

        # compute solution, and iterate until optimal k
        def objective(k):
            beta = np.linalg.pinv(APA + k * K0) @ A.T @ P @ Y
            residual = np.linalg.norm(Y - A @ beta)
            parameter_size = np.linalg.norm(k * K0 * beta)
            # parameter_size = np.linalg.norm(beta)
            score = residual + parameter_size
            return score

        res = minimize(objective, [k], tol=1e-16, bounds=Bounds(0, np.inf))
        # res = minimize(objective, [k], tol=1E-8)
        print(f"Total iterations: {res.nit} \t k={res.x[0]}")
        k = res.x[0]
        return np.linalg.pinv(APA + k * K0) @ A.T @ P @ Y

    def update(self, rVec, aVec, R=None):
        self.rVec1D = rVec.reshape((-1,))
        self.aVec1D = aVec.reshape((-1,))
        self.P = len(self.rVec1D)

        M = populate_M(
            self.rVec1D,
            self.A,
            self.n1,
            self.n2,
            self.N,
            self.a,
            self.mu,
            self.remove_deg,
        )
        if R is None:
            R = np.eye(len(M))
        results = self.compute_coefficients(M, self.aVec1D, R)
        return results


class AnalyzeRegression:
    def __init__(self, true_Clm, true_Slm, pred_Clm, pred_Slm):
        self.true_Clm = true_Clm
        self.true_Slm = true_Slm
        self.pred_Clm = pred_Clm
        self.pred_Slm = pred_Slm

    def compute_degree_variance(self, C_lm, S_lm):
        N = len(C_lm)
        rms = np.zeros((N,))
        for i in range(N):
            rms_sum = 0.0
            for j in range(i + 1):
                rms_sum += C_lm[i, j] ** 2 + S_lm[i, j] ** 2
            rms_sum *= 1 / (2 * i + 1)
            rms[i] = np.sqrt(rms_sum)
        return rms

    def plot_coef_rms(self, C_lm, S_lm):
        N = len(C_lm)
        degrees = np.arange(0, N)
        rms = self.compute_degree_variance(C_lm, S_lm)
        plt.semilogy(degrees, rms)
        plt.xlim([2, None])


def print_coefficients(C_lm, S_lm):
    for i in range(len(C_lm)):
        for j in range(i + 1):
            print(
                f"({i},{j}): {sigfig.round(float(C_lm[i,j]),sigfigs=2, notation='scientific')} \t {sigfig.round(float(S_lm[i,j]),sigfigs=2,notation='scientific')}",
            )


def simple_experiment():
    import time

    from GravNN.CelestialBodies.Planets import Earth
    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
    from GravNN.Trajectories import RandomDist

    max_true_deg = 30
    regress_deg = 16
    remove_deg = 0
    # solver_algorithm='kaula'
    solver_algorithm = "single_parameter"
    # solver_algorithm='least_squares'

    sigma = 1e-8  # assume no measurement noise -- this might need to be smaller

    planet = Earth()
    sh_EGM2008 = SphericalHarmonics(planet.sh_file, regress_deg)

    # trajectory = DHGridDist(planet, sh_EGM2008.radEquator, regress_deg+1)
    trajectory = RandomDist(planet, [planet.radius, planet.radius + 420], 1000)
    x, a, u = get_sh_data(
        trajectory,
        planet.sh_file,
        max_deg=max_true_deg,
        deg_removed=remove_deg,
    )

    regressor = XuLS(regress_deg, planet, remove_deg, solver_algorithm)
    start = time.time()
    R = sigma**2 * np.eye(len(x))
    results = regressor.update(x, a, R)
    C_lm, S_lm = format_coefficients(results, regress_deg, remove_deg)
    print(time.time() - start)

    k = len(C_lm)
    true_C_lm = sh_EGM2008.C_lm[:k, :k]
    true_S_lm = sh_EGM2008.S_lm[:k, :k]

    C_lm_error = (true_C_lm - C_lm) / true_C_lm * 100
    S_lm_error = (true_S_lm - S_lm) / true_S_lm * 100

    C_lm_error = np.nan_to_num(C_lm_error, posinf=0, neginf=0)
    S_lm_error = np.nan_to_num(S_lm_error, posinf=0, neginf=0)

    print_coefficients(C_lm_error, S_lm_error)

    analyzer = AnalyzeRegression(true_C_lm, true_S_lm, C_lm, S_lm)
    analyzer.plot_coef_rms(true_C_lm, true_S_lm)
    analyzer.plot_coef_rms(C_lm, S_lm)
    analyzer.plot_coef_rms(C_lm - true_C_lm, S_lm - true_S_lm)
    plt.show()


if __name__ == "__main__":
    simple_experiment()
