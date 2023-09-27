import numpy as np
from numba import njit


@njit(cache=True)
def getK(l):
    result = 1.0 if (l == 0) else 2.0
    return result


@njit(cache=True)
def compute_A(A, n1, n2, u):
    # Eq 23
    for n in range(1, len(A)):
        A[n, n - 1] = np.sqrt(((2.0 * n) * getK(n - 1.0)) / getK(n)) * A[n, n] * u

    for m in range(0, len(A)):
        for n in range(m + 2, len(A)):
            A[n, m] = u * n1[n][m] * A[n - 1, m] - n2[n][m] * A[n - 2, m]

    return A


@njit(cache=True)
def compute_euler(N, a, mu, rMag, s, t):
    rE = np.zeros((N + 2,))
    iM = np.zeros((N + 2,))
    rho = np.zeros((N + 3,))

    # Eq 24
    rE[0] = 1  # cos(m*lambda)*cos(m*alpha)
    iM[0] = 0  # sin(m*lambda)*cos(m*alpha)
    for m in range(1, len(rE)):
        rE[m] = s * rE[m - 1] - t * iM[m - 1]
        iM[m] = s * iM[m - 1] + t * rE[m - 1]

    # Eq 26 and 26a
    beta = a / rMag
    rho[0] = mu / rMag
    rho[1] = rho[0] * beta
    for n in range(2, len(rho)):
        rho[n] = beta * rho[n - 1]
    return rE, iM, rho


@njit(cache=True)  # , parallel=True)
def populate_M(rVec1D, A, n1, n2, N, a, mu, remove_deg):
    P = len(rVec1D)
    Q = N + 1  # Total Indicies Needed to store all coefficients
    if remove_deg:
        M = np.zeros((P, Q * (Q + 1) - 2 * (2 + 1)))
    else:
        M = np.zeros((P, Q * (Q + 1)))

    for p in range(0, int(P / 3)):
        rVal = rVec1D[3 * p : 3 * (p + 1)]
        rMag = np.linalg.norm(rVal)
        x, y, z = rVec1D[3 * p : 3 * (p + 1)]
        s, t, u = rVal / rMag

        # populate variables
        A = compute_A(A, n1, n2, u)
        rE, iM, rho = compute_euler(N, a, mu, rMag, s, t)

        # NOTE: NO ESTIMATION OF C00, C10, C11 -- THESE ARE DETERMINED ALREADY
        start = 2 if remove_deg else 0
        for n in range(start, N + 1):
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

                # Seems representative of the computed model
                f_Cnm_1 = (
                    (rho[n + 1] / a)
                    * m
                    * (A[n, m] * rTerm - s * c2 * A[n + 1, m + 1] * rE[m])
                )
                f_Cnm_2 = (
                    (rho[n + 1] / a)
                    * m
                    * (-1 * A[n, m] * iTerm - t * c2 * A[n + 1, m + 1] * rE[m])
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

                f_Snm_1 = (
                    (rho[n + 1] / a)
                    * m
                    * (A[n, m] * iTerm - s * c2 * A[n + 1, m + 1] * iM[m])
                )
                f_Snm_2 = (
                    (rho[n + 1] / a)
                    * m
                    * (A[n, m] * rTerm - t * c2 * A[n + 1, m + 1] * iM[m])
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

                # idx = n - 2 # The M matrix excludes columns for C00, C10, C11 so we need to subtract 2 from the current degree for proper indexing
                # idx = n
                if remove_deg:
                    degIdx = n * (n + 1) - (2 * (2 + 1))
                else:
                    degIdx = n * (n + 1)

                M[3 * p + 0, degIdx + 2 * m + 0] = f_Cnm_1  # X direction
                M[3 * p + 0, degIdx + 2 * m + 1] = f_Snm_1
                M[3 * p + 1, degIdx + 2 * m + 0] = f_Cnm_2  # Y direction
                M[3 * p + 1, degIdx + 2 * m + 1] = f_Snm_2
                M[3 * p + 2, degIdx + 2 * m + 0] = f_Cnm_3  # Z direction
                M[3 * p + 2, degIdx + 2 * m + 1] = f_Snm_3

    return M


def format_coefficients(coefficients, remove_deg):
    coefficients = coefficients.reshape((-1, 2))

    l = 0
    m = 0

    if remove_deg:
        C00_row = np.array([[1.0, 0.0]])
        C10_row = np.array([[0.0, 0.0]])
        C11_row = np.array([[0.0, 0.0]])
        coefficients = np.concatenate([C00_row, C10_row, C11_row, coefficients], axis=0)
    coef_final = np.zeros((len(coefficients), 4))

    for i in range(0, len(coefficients)):
        coef_final[i, 0] = int(l)
        coef_final[i, 1] = int(m)
        coef_final[i, 2] = coefficients[i, 0]
        coef_final[i, 3] = coefficients[i, 1]

        if m < l:
            m += 1
        else:
            l += 1
            m = 0

    return coef_final


def iterate_lstsq(M, aVec, iterations):
    results = np.linalg.lstsq(M, aVec)[0]
    delta_a = aVec - np.dot(M, results)
    for i in range(iterations):
        delta_coef = np.linalg.lstsq(M, delta_a)[0]
        results -= delta_coef
        delta_a = aVec - np.dot(M, results)
    return results


class Regression:
    def __init__(self, max_deg, planet, rVec, aVec):
        self.N = max_deg  # Degree
        self.a = planet.radius
        self.mu = planet.mu
        self.rVec1D = rVec.reshape((-1,))
        self.aVec1D = aVec.reshape((-1,))

        self.N = self.N

        self.P = len(self.rVec1D)
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

            np.zeros((i + 1,)).tolist()
            np.zeros((i + 1,)).tolist()

            for m in range(0, i + 1):  # Check the plus one
                if i >= m + 2:
                    self.n1[i, m] = np.sqrt(
                        ((2.0 * i + 1.0) * (2.0 * i - 1.0)) / ((i - m) * (i + m)),
                    )
                    self.n2[i, m] = np.sqrt(
                        ((i + m - 1.0) * (2.0 * i + 1.0) * (i - m - 1.0))
                        / ((i + m) * (i - m) * (2.0 * i - 3.0)),
                    )
            # self.n1.append(n1Row)
            # self.n2.append(n2Row)

    def perform_regression(self, remove_deg=False):
        M = populate_M(
            self.rVec1D,
            self.A,
            self.n1,
            self.n2,
            self.N,
            self.a,
            self.mu,
            remove_deg,
        )
        results = iterate_lstsq(M, self.aVec1D, 1)

        self.coef = format_coefficients(results, remove_deg)

        return self.coef

    def save(self, file_name):
        header_data = np.array([[self.a, self.mu]])

        fmt = "%.12E"

        np.savetxt(file_name, self.coef, delimiter=",", fmt=fmt)

        with open(file_name, "r") as f:
            content = f.read()  # .splitlines()

        np.savetxt(file_name, header_data, delimiter=",", fmt=fmt)

        with open(file_name, "r") as f:
            header_content = f.read()  # .splitlines()

        with open(file_name, "w", newline="") as f:
            f.write(header_content)
            f.write(content)


def main():
    import time

    from GravNN.CelestialBodies.Planets import Earth
    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
    from GravNN.Trajectories import DHGridDist

    max_true_deg = 4
    regress_deg = 4
    sh_EGM2008 = SphericalHarmonics(
        "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\GravNN\\Files\\GravityModels\\EGM2008_to2190_TideFree_E.txt",
        regress_deg,
    )

    planet = Earth()
    # trajectory = DHGridDist(planet, planet.radius, 180)
    trajectory = DHGridDist(planet, sh_EGM2008.radEquator, 360)

    remove_deg = True
    if remove_deg:
        x, a, u = get_sh_data(trajectory, planet.sh_file, max_true_deg, 0)
    else:
        x, a, u = get_sh_data(trajectory, planet.sh_file, max_true_deg, -1)

    regressor = Regression(regress_deg, planet, x, a)
    start = time.time()
    regressor.perform_regression(remove_deg)
    print(time.time() - start)
    regressor.save(
        "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\GravNN\\Files\\GravityModels\\Regressed\\some.csv",
    )
    # print(coefficients)

    SphericalHarmonics(
        "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\GravNN\\Files\\GravityModels\\Regressed\\truth.csv",
        regress_deg,
    )
    sh_regress = SphericalHarmonics(
        "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\GravNN\\Files\\GravityModels\\Regressed\\some.csv",
        regress_deg,
    )

    # print(sh_EGM2008.C_lm[:,0])
    print(
        np.round(
            (sh_regress.C_lm[:, 0] - sh_EGM2008.C_lm[:, 0][:-2])
            / sh_EGM2008.C_lm[:, 0][:-2]
            * 100,
        ),
    )
    print(
        np.round(
            (sh_regress.C_lm[:, 1] - sh_EGM2008.C_lm[:, 1][:-2])
            / sh_EGM2008.C_lm[:, 1][:-2]
            * 100,
        ),
    )
    print(
        np.round(
            (sh_regress.C_lm[:, 2] - sh_EGM2008.C_lm[:, 2][:-2])
            / sh_EGM2008.C_lm[:, 2][:-2]
            * 100,
        ),
    )
    print(
        np.round(
            (sh_regress.C_lm[:, 3] - sh_EGM2008.C_lm[:, 3][:-2])
            / sh_EGM2008.C_lm[:, 3][:-2]
            * 100,
        ),
    )

    errors = np.round(
        (sh_regress.C_lm.reshape((-1)) - sh_EGM2008.C_lm[:-2, :-2].reshape((-1)))
        / sh_EGM2008.C_lm[:-2, :-2].reshape((-1))
        * 100,
        1,
    )
    print(errors)
    # l = [x for x in errors if ~np.isnan(x)]
    # print(l)
    # assert(np.allclose(sh_test.C_lm ,sh_regress.C_lm,atol=1E-11))
    # assert(np.allclose(sh_test.S_lm, sh_regress.S_lm,atol=1E-11))

    # Load test data, and check for differences
    # Move the positions to be the rvecto


if __name__ == "__main__":
    main()
