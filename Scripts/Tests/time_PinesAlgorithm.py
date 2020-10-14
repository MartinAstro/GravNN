
import os, sys
import numpy as np
import GravNN

from GravNN.GravityModels.PinesAlgorithm import PinesAlgorithm
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
def make_2D_array(lis):
    """Funciton to get 2D array from a list of lists
    """
    n = len(lis)
    lengths = np.array([len(x) for x in lis])
    max_len = np.max(lengths)
    arr = np.zeros((n, max_len))

    for i in range(n):
        arr[i, :lengths[i]] = lis[i]
    return arr

gravityModel = SphericalHarmonics(GravNN.__path__[0]  + "/Files/GravityModels/GGM03S.txt", 100)
planet = Earth(gravityModel)

N = 100
positions = np.random.uniform(1, 1.5, (3*N))*planet.radius

degree = 100
cbar = make_2D_array(gravityModel.C_lm)
sbar = make_2D_array(gravityModel.S_lm)
pines = PinesAlgorithm(planet.radius, planet.mu, degree, cbar, sbar)
pines.compute_acc(positions)

def timing():

    import timeit
    print(timeit.timeit('pines.compute_acc(positions)', globals=globals(), number=100))
    #print(timeit.timeit('pines.compute_acc_jit(positions)', globals=globals(), number=100))

if __name__ == "__main__":
    #pines.compute_acc_jit(positions)
    timing()