
import numpy as np
from numpy.random import seed

from Trajectories.UniformDist import UniformDist
from CelestialBodies.Planets import Earth
from AccelerationAlgs.SphericalHarmonics import SphericalHarmonics

import os, sys
sys.path.append(os.path.dirname(__file__) + "/build/PinesAlgorithm/")
import PinesAlgorithm

def main():
    degree = 4
    planet = Earth()
    planet.loadSH()
    trajectory = UniformDist(planet, planet.geometry.radius, 100)

    positions = np.reshape(trajectory.positions, (len(trajectory.positions)*3))
    pines = PinesAlgorithm.PinesAlgorithm(planet.geometry.radius, planet.grav_info.mu, degree)
    accelerations = pines.compute_acc(positions, planet.grav_info.SH.C_lm, planet.grav_info.SH.S_lm)
    accelerations = np.reshape(np.array(accelerations), (int(len(np.array(accelerations))/3), 3))
    print(accelerations)




if __name__ == "__main__":
    main()