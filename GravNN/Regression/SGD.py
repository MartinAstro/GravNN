import numpy as np

from GravNN.Regression.utils import (
    getK,
)


class SGD:
    def __init__(self, radius, degree):
        self.radius = radius
        self.degree = degree

        # compute number of coefficients
        nCS = 0
        nC = 0
        nC_ii = 3
        nS_ii = 2
        for ii in range(2, degree + 1):
            nC = nC + nC_ii
            nCS = nCS + nC_ii + nS_ii
            nC_ii += 1
            nS_ii += 1

        aBar, n1, n2, nq1, nq2 = self.initialize_pines_params(degree)

        self.aBar = aBar
        self.n1 = n1
        self.n2 = n2
        self.nq1 = nq1
        self.nq2 = nq2

    def getK(self, x):
        return 1.0 if (x == 0) else 2.0

    def initialize_pines_params(self, degree):
        # preallocate variables
        aBar = np.zeros((degree + 2, degree + 2))
        n1 = np.zeros((degree + 2, degree + 2))
        n2 = np.zeros((degree + 2, degree + 2))
        nq1 = np.zeros((degree + 2, degree + 2))
        nq2 = np.zeros((degree + 2, degree + 2))

        # loop through degree
        for ii in range(0, degree + 2):
            # fill diagonal terms
            if ii == 0:
                aBar[ii, ii] = 1
            else:
                aBar[ii, ii] = (
                    np.sqrt((2 * ii + 1) * getK(ii) / (2 * ii * getK(ii - 1)))
                    * aBar[ii - 1, ii - 1]
                )

            # loop through order
            for jj in range(0, ii + 1):
                if ii >= jj + 2:
                    n1[ii, jj] = np.sqrt(
                        (2 * ii + 1) * (2 * ii - 1) / ((ii - jj) * (ii + jj)),
                    )
                    n2[ii, jj] = np.sqrt(
                        (ii + jj - 1)
                        * (2 * ii + 1)
                        * (ii - jj - 1)
                        / ((ii + jj) * (ii - jj) * (2 * ii - 3)),
                    )

        for ii in range(0, degree + 1):
            for jj in range(0, ii + 1):
                if jj < ii:
                    nq1[ii, jj] = np.sqrt(
                        (ii - jj) * getK(jj) * (ii + jj + 1) / getK(jj + 1),
                    )
                nq2[ii, jj] = np.sqrt(
                    (ii + jj + 2)
                    * (ii + jj + 1)
                    * (2 * ii + 1)
                    * getK(jj)
                    / ((2 * ii + 3) * getK(jj + 1)),
                )

        return aBar, n1, n2, nq1, nq2

    def computePinesGravitySensitivity(self, posBatch, degree, mu, rE):
        # initialize sensitivity matrix
        nBatch = len(posBatch)
        dadCS = np.zeros((3 * nBatch, nCS))

        # loop through batch
        for kk in range(nBatch):
            # retrieve current position
            pos_kk = posBatch[kk][0:3]

            # compute Pines coordinates
            r = np.linalg.norm(pos_kk)
            s = pos_kk[0] / r
            t = pos_kk[1] / r
            u = pos_kk[2] / r

            for ii in range(1, degree + 2):
                # compute low diagonal terms
                self.aBar[ii, ii - 1] = (
                    np.sqrt((2 * ii) * getK(ii - 1) / getK(ii)) * self.aBar[ii, ii] * u
                )

            # compute lower terms of A_bar
            rEVec = np.zeros(degree + 2)
            iM = np.zeros(degree + 2)
            for jj in range(0, degree + 2):
                for ii in range(jj + 2, degree + 2):
                    self.aBar[ii, jj] = (
                        u * self.n1[ii, jj] * self.aBar[ii - 1, jj]
                        - self.n2[ii, jj] * self.aBar[ii - 2, jj]
                    )

                if jj == 0:
                    rEVec[jj] = 1.0
                    iM[jj] = 0.0
                else:
                    rEVec[jj] = s * rEVec[jj - 1] - t * iM[jj - 1]
                    iM[jj] = s * iM[jj - 1] + t * rEVec[jj - 1]

            # define variables
            rho = rE / r
            rhoVec = np.zeros(degree + 2)
            rhoVec[0] = mu / r
            rhoVec[1] = rhoVec[0] * rho
            rhoVec[2] = rhoVec[1] * rho

            # set counter
            cont_C = 0
            cont_S = 0

            # loop through degree
            for ii in range(2, degree + 1):
                # add rhoVec term
                rhoVec[ii + 1] = rho * rhoVec[ii]

                # loop through order
                for jj in range(0, ii + 1):
                    if jj == 0:
                        dsa1dCij = 0.0
                        dsa1dSij = 0.0
                        dsa2dCij = 0.0
                        dsa2dSij = 0.0
                    else:
                        dsa1dCij = jj * self.aBar[ii, jj] * rEVec[jj - 1]
                        dsa1dSij = jj * self.aBar[ii, jj] * iM[jj - 1]
                        dsa2dCij = jj * self.aBar[ii, jj] * (-iM[jj - 1])
                        dsa2dSij = jj * self.aBar[ii, jj] * rEVec[jj - 1]

                    if jj < ii:
                        dsa3dCij = self.nq1[ii, jj] * self.aBar[ii, jj + 1] * rEVec[jj]
                        dsa3dSij = self.nq1[ii, jj] * self.aBar[ii, jj + 1] * iM[jj]
                    else:
                        dsa3dCij = 0.0
                        dsa3dSij = 0.0
                    dsa4dCij = self.nq2[ii, jj] * self.aBar[ii + 1, jj + 1] * rEVec[jj]
                    dsa4dSij = self.nq2[ii, jj] * self.aBar[ii + 1, jj + 1] * iM[jj]

                    da0dCij = (dsa1dCij - s * dsa4dCij) * rhoVec[ii + 1] / rE
                    da0dSij = (dsa1dSij - s * dsa4dSij) * rhoVec[ii + 1] / rE
                    da1dCij = (dsa2dCij - t * dsa4dCij) * rhoVec[ii + 1] / rE
                    da1dSij = (dsa2dSij - t * dsa4dSij) * rhoVec[ii + 1] / rE
                    da2dCij = (dsa3dCij - u * dsa4dCij) * rhoVec[ii + 1] / rE
                    da2dSij = (dsa3dSij - u * dsa4dSij) * rhoVec[ii + 1] / rE

                    dadCS[3 * kk : 3 * (kk + 1), cont_C] = np.array(
                        [da0dCij, da1dCij, da2dCij],
                    )
                    cont_C += 1
                    if jj > 0:
                        dadCS[3 * kk : 3 * (kk + 1), nC + cont_S] = np.array(
                            [da0dSij, da1dSij, da2dSij],
                        )
                        cont_S += 1

        return dadCS

    def update(self, positions):
        self.computePinesGravitySensitivity(positions)

        w - learning_rate * g

        return


##################


if __name__ == "__main__":
    planet = Earth()
    degree = 10
    radius = planet.radius

    optimizer = SGD(radius, degree)
