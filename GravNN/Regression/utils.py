import os

import numpy as np
from numba import njit


@njit(cache=True)
def getK(l):  # noqa: E741
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


def format_coefficients(coefficients, regress_deg, remove_deg):
    coefficients = coefficients.reshape((-1, 2))

    N = regress_deg
    M = remove_deg
    C_lm = np.zeros((N + 1, N + 1))
    S_lm = np.zeros((N + 1, N + 1))

    if M != -1:
        C_lm[0, 0] = 1.0

    k = 0
    for i in range(M + 1, N + 1):
        for j in range(i + 1):
            C_lm[i, j] = coefficients[k, 0]
            S_lm[i, j] = coefficients[k, 1]
            k += 1

    return C_lm, S_lm


def populate_removed_degrees(C_lm_hat, S_lm_hat, C_lm, S_lm, remove_deg):
    for i in range(0, remove_deg + 1):
        for j in range(i + 1):
            C_lm_hat[i, j] = C_lm[i, j]
            S_lm_hat[i, j] = S_lm[i, j]

    return C_lm_hat, S_lm_hat


def save(file_name, planet, C_lm, S_lm):
    header_data = "%.12E, %.12E \n" % (planet.radius, planet.mu)
    data = ""
    for i in range(len(C_lm)):
        for j in range(i + 1):
            data += "%d, %d, %.12E, %.12E \n" % (i, j, C_lm[i, j], S_lm[i, j])

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", newline="") as f:
        f.write(header_data)
        f.write(data)


class RegressSolution:
    def __init__(self, results, regress_deg, remove_deg, planet):
        C_lm, S_lm = format_coefficients(results, regress_deg, remove_deg)
        self.C_lm = C_lm
        self.S_lm = S_lm
        self.planet = planet


def preprocess_data(x_dumb, a_dumb, acc_noise, pos_noise):
    x_dumb = np.array(x_dumb)
    a_dumb = np.array(a_dumb)

    x_error = pos_noise * np.random.normal(0, 1, size=np.shape(x_dumb))
    x_dumb += x_error

    # (Optionally) Add noise
    a_mag = np.linalg.norm(a_dumb, axis=1).reshape(len(a_dumb), 1)
    a_unit = np.random.uniform(-1, 1, size=np.shape(a_dumb))
    a_unit = a_unit / np.linalg.norm(a_unit, axis=1).reshape(len(a_unit), 1)
    a_error = acc_noise * a_mag * a_unit  # 10% of the true magnitude
    a_dumb = a_dumb + a_error

    return x_dumb, a_dumb


def append_data(x_train, y_train, x, y):
    try:
        for i in range(len(x)):
            x_train.append(x[i])
            y_train.append(y[i])
    except Exception:
        x_train = np.concatenate((x_train, x))
        y_train = np.concatenate((y_train, y))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    return x_train, y_train
