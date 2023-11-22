import multiprocessing as mp
from functools import partial

import numpy as np
from numba import njit

from GravNN.Support.slurm_utils import get_available_cores


def getK(x):
    return 1.0 if (x == 0) else 2.0


def compute_n_matrices(N):
    n1 = np.ones((N + 2, N + 2)) * np.nan
    n2 = np.ones((N + 2, N + 2)) * np.nan
    n1q = np.ones((N + 2, N + 2)) * np.nan
    n2q = np.ones((N + 2, N + 2)) * np.nan

    for l in range(0, N + 2):  # noqa: E741
        for m in range(0, l + 1):
            if l >= m + 2:
                n1[l][m] = np.sqrt(
                    ((2.0 * l + 1.0) * (2.0 * l - 1.0)) / ((l - m) * (l + m)),
                )
                n2[l][m] = np.sqrt(
                    ((l + m - 1.0) * (2.0 * l + 1.0) * (l - m - 1.0))
                    / ((l + m) * (l - m) * (2.0 * l - 3.0)),
                )
            if l < N + 1:
                if m < l:  # this may need to also ensure that l < N+1
                    n1q[l][m] = np.sqrt(
                        ((l - m) * getK(m) * (l + m + 1.0)) / getK(m + 1),
                    )
                n2q[l][m] = np.sqrt(
                    ((l + m + 2.0) * (l + m + 1.0) * (2.0 * l + 1.0) * getK(m))
                    / ((2.0 * l + 3.0) * getK(m + 1.0)),
                )
    return n1, n2, n1q, n2q


def compute_acceleration(positions, N, mu, a, n1, n2, n1q, n2q, cbar, sbar):
    acc = np.zeros(positions.shape)
    for i in range(0, int(len(positions) / 3)):
        r = np.linalg.norm(positions[3 * i : 3 * (i + 1)])
        [s, t, u] = positions[3 * i : 3 * (i + 1)] / r

        rE = np.zeros((N + 2,))
        iM = np.zeros((N + 2,))

        rhol = np.zeros((N + 2,))

        aBar = np.zeros((N + 2, N + 2))
        aBar[0, 0] = 1.0

        rho = a / r
        rhol[0] = mu / r
        rhol[1] = rhol[0] * rho

        for l in range(1, N + 2):  # noqa: E741
            aBar[l][l] = (
                np.sqrt(((2.0 * l + 1.0) * getK(l)) / ((2.0 * l * getK(l - 1))))
                * aBar[l - 1][l - 1]
            )
            aBar[l][l - 1] = np.sqrt((2.0 * l) * getK(l - 1) / getK(l)) * aBar[l][l] * u

        for m in range(0, N + 2):
            for l in range(m + 2, N + 2):  # noqa: E741
                aBar[l][m] = u * n1[l][m] * aBar[l - 1][m] - n2[l][m] * aBar[l - 2][m]
            rE[m] = 1.0 if m == 0 else s * rE[m - 1] - t * iM[m - 1]
            iM[m] = 0.0 if m == 0 else s * iM[m - 1] + t * rE[m - 1]

        a1, a2, a3, a4 = 0.0, 0.0, 0.0, 0.0
        for l in range(1, N + 1):  # noqa: E741
            rhol[l + 1] = rho * rhol[l]
            sum_a1, sum_a2, sum_a3, sum_a4 = 0.0, 0.0, 0.0, 0.0
            for m in range(0, l + 1):
                D = cbar[l][m] * rE[m] + sbar[l][m] * iM[m]
                E = 0.0 if m == 0 else cbar[l][m] * rE[m - 1] + sbar[l][m] * iM[m - 1]
                F = 0.0 if m == 0 else sbar[l][m] * rE[m - 1] - cbar[l][m] * iM[m - 1]

                sum_a1 += m * aBar[l][m] * E
                sum_a2 += m * aBar[l][m] * F

                if m < l:
                    sum_a3 += n1q[l][m] * aBar[l][m + 1] * D
                sum_a4 += n2q[l][m] * aBar[l + 1][m + 1] * D
            a1 += rhol[l + 1] / a * sum_a1
            a2 += rhol[l + 1] / a * sum_a2
            a3 += rhol[l + 1] / a * sum_a3
            a4 -= rhol[l + 1] / a * sum_a4
        a4 -= rhol[1] / a

        acc[3 * i : 3 * (i + 1)] = np.array([a1, a2, a3]) + np.array([s, t, u]) * a4
    return acc


# @njit(cache=True)
# def compute_acc(positions, N, mu, a, n1, n2, n1q, n2q, cbar, sbar):
#     acc = np.zeros(positions.shape)
#     N_total = int(len(positions) / 3)
#     potential = np.zeros((N_total,))
#     if N == -1:
#         return (acc, potential)
#     for i in prange(0, N_total):
#         results = compute_acc_thread(
#             positions[3 * i : 3 * (i + 1)],
#             N,
#             mu,
#             a,
#             n1,
#             n2,
#             n1q,
#             n2q,
#             cbar,
#             sbar,
#         )
#         acc[3 * i : 3 * (i + 1)] = results[0]
#         potential[i] = results[1]
#     return (acc, potential)


def compute_acc(positions, N, mu, a, n1, n2, n1q, n2q, cbar, sbar):
    acc = np.zeros(positions.shape)
    N_total = int(len(positions) / 3)
    potential = np.zeros((N_total,))
    if N == -1:
        return (acc, potential)

    compute_acc_partial = partial(
        compute_acc_thread,
        N=N,
        mu=mu,
        a=a,
        n1=n1,
        n2=n2,
        n1q=n1q,
        n2q=n2q,
        cbar=cbar,
        sbar=sbar,
    )
    positions_Nx3 = positions.reshape((-1, 3))
    if len(positions_Nx3) == 1:
        results = [compute_acc_partial(positions_Nx3[0])]
    else:
        cores = get_available_cores()
        with mp.Pool(processes=cores) as pool:
            results = pool.map(compute_acc_partial, positions_Nx3)

    for i, result in enumerate(results):
        acc_output = result[0]
        pot_output = result[1]

        acc[3 * i : 3 * (i + 1)] = acc_output
        potential[i] = pot_output
    return (acc, potential)


@njit(cache=True, parallel=False)
def compute_acc_thread(position, N, mu, a, n1, n2, n1q, n2q, cbar, sbar):
    np.zeros(position.shape)
    potential = 0.0
    r = np.linalg.norm(position)
    [s, t, u] = position / r

    rE = np.zeros((N + 2,))
    iM = np.zeros((N + 2,))

    rhol = np.zeros((N + 2,))

    aBar = np.zeros((N + 2, N + 2))
    aBar[0, 0] = 1.0

    rho = a / r
    rhol[0] = mu / r
    rhol[1] = rhol[0] * rho

    for l in range(1, N + 2):  # noqa: E741
        aBar[l][l] = (
            np.sqrt(((2.0 * l + 1.0) * getK(l)) / ((2.0 * l * getK(l - 1))))
            * aBar[l - 1][l - 1]
        )
        aBar[l][l - 1] = np.sqrt((2.0 * l) * getK(l - 1) / getK(l)) * aBar[l][l] * u

    for m in range(0, N + 2):
        for l in range(m + 2, N + 2):  # noqa: E741
            aBar[l][m] = u * n1[l][m] * aBar[l - 1][m] - n2[l][m] * aBar[l - 2][m]
        rE[m] = 1.0 if m == 0 else s * rE[m - 1] - t * iM[m - 1]
        iM[m] = 0.0 if m == 0 else s * iM[m - 1] + t * rE[m - 1]

    a1, a2, a3, a4 = 0.0, 0.0, 0.0, 0.0
    for l in range(1, N + 1):  # noqa: E741
        rhol[l + 1] = rho * rhol[l]
        sum_a1, sum_a2, sum_a3, sum_a4 = 0.0, 0.0, 0.0, 0.0
        for m in range(0, l + 1):
            D = cbar[l][m] * rE[m] + sbar[l][m] * iM[m]
            E = 0.0 if m == 0 else cbar[l][m] * rE[m - 1] + sbar[l][m] * iM[m - 1]
            F = 0.0 if m == 0 else sbar[l][m] * rE[m - 1] - cbar[l][m] * iM[m - 1]

            sum_a1 += m * aBar[l][m] * E
            sum_a2 += m * aBar[l][m] * F

            if m < l:
                sum_a3 += n1q[l][m] * aBar[l][m + 1] * D
            sum_a4 += n2q[l][m] * aBar[l + 1][m + 1] * D

            potential += rhol[l] * aBar[l][m] * D
        a1 += rhol[l + 1] / a * sum_a1
        a2 += rhol[l + 1] / a * sum_a2
        a3 += rhol[l + 1] / a * sum_a3
        a4 -= rhol[l + 1] / a * sum_a4
    a4 -= rhol[1] / a

    # The prior loop doesn't account for the l=0 index
    potential += rhol[0] * aBar[0][0] * (cbar[0][0] * rE[0] + sbar[0][0] * iM[0])

    # Note that the original paper computes U and F=dU (as opposed to U and F=-dU)
    # Consequently, F in the paper is actually equal to -a, but all of my calculations
    # used the assumption that F = a so instead of changing multiplying the acceleration
    # generated by -1, we multiply the potential by -1 because it is used in
    # significantly fewer places and then reconciles the relationship with the
    # produced acceleration.
    return (np.array([a1, a2, a3]) + np.array([s, t, u]) * a4, -potential)


getK = njit(getK, cache=True)
compute_n_matrices = njit(compute_n_matrices, cache=True)
# compute_acc_jit = njit(compute_acc, parallel=False, cache=True)
# compute_acc_parallel = njit(compute_acc, parallel=True, cache=True)
# compute_acc_thread = njit(compute_acc_thread, cache=True)

# compute_acc_jit = compute_acc
