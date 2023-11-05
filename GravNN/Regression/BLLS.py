import numpy as np
from SHRegression import SHRegression

from GravNN.Regression.utils import format_coefficients


def iterate_lstsq(M, aVec, iterations):
    results = np.linalg.lstsq(M, aVec)[0]
    delta_a = aVec - np.dot(M, results)
    for i in range(iterations):
        delta_coef = np.linalg.lstsq(M, delta_a)[0]
        results -= delta_coef
        delta_a = aVec - np.dot(M, results)
    return results


class BLLS:
    def __init__(self, max_deg, planet, remove_deg=-1):
        self.N = max_deg  # Degree
        self.remove_deg = remove_deg

        self.SHRegressor = SHRegression(max_deg, planet.radius, planet.mu)

    def update(self, rVec, aVec, iterations=5):
        self.rVec1D = rVec.reshape((-1,))
        self.aVec1D = aVec.reshape((-1,))
        self.P = len(self.rVec1D)

        M = self.SHRegressor.populate_M(
            self.rVec1D,
            self.remove_deg,
        )
        results = iterate_lstsq(M, self.aVec1D, iterations)
        return results


class BLLS_PM:
    def __init__(self, max_deg, planet, remove_deg=-1):
        self.N = max_deg  # Degree
        self.a = planet.radius
        self.mu = planet.mu
        self.remove_deg = remove_deg

    def update(self, rVec, aVec, iterations=5):
        r = np.linalg.norm(rVec, axis=1)
        r_hat = rVec / r.reshape((-1, 1))
        a_pm = -self.mu * r_hat / r.reshape((-1, 1)) ** 2
        da_dmu = a_pm / self.mu
        M = da_dmu.reshape((-1, 1))

        self.rVec1D = rVec.reshape((-1,))
        self.aVec1D = aVec.reshape((-1,))
        self.P = len(self.rVec1D)

        results = iterate_lstsq(M, self.aVec1D, iterations)
        results /= self.mu
        return results


def main():
    import time

    from GravNN.CelestialBodies.Planets import Earth
    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
    from GravNN.Trajectories import DHGridDist

    max_true_deg = 10
    regress_deg = 5
    remove_deg = -1
    # remove_deg = 0 # C20 is very close, C22 isn't that close

    planet = Earth()
    sh_EGM2008 = SphericalHarmonics(planet.sh_file, regress_deg)

    trajectory = DHGridDist(planet, sh_EGM2008.radEquator, 5)
    # trajectory = RandomDist(planet, [planet.radius, planet.radius+420], 1000)

    x, a, u = get_sh_data(
        trajectory,
        planet.sh_file,
        max_deg=max_true_deg,
        deg_removed=remove_deg,
    )

    regressor = BLLS(regress_deg, planet, remove_deg)
    start = time.time()
    results = regressor.update(x, a)
    C_lm, S_lm = format_coefficients(results, regress_deg, remove_deg)
    print(time.time() - start)

    k = len(C_lm)
    C_lm_true = sh_EGM2008.C_lm[:k, :k]
    S_lm_true = sh_EGM2008.S_lm[:k, :k]

    C_lm_error = (C_lm_true - C_lm) / C_lm_true * 100
    S_lm_error = (S_lm_true - S_lm) / S_lm_true * 100

    print(np.array2string(C_lm_error, precision=0))
    print(np.array2string(S_lm_error, precision=0))

    # regressor.save('C:\\Users\\John\\Documents\\Research\\ML_Gravity\\GravNN\\Files\\GravityModels\\Regressed\\some.csv')
    # print(coefficients)


if __name__ == "__main__":
    main()
