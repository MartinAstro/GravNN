
import numpy as np
from GravNN.Support.Regression import regress
from GravNN.Trajectories.UniformDist import UniformDist
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics

def main():
    planet = Earth()
    trajectory = UniformDist(planet, planet.radius, 100)
    #trajectory = RandomDist(planet, [planet.radius+100, planet.radius+500], 10000) 

    # Full fidelity model
    max_deg = 3
    gravityModel = SphericalHarmonics(planet.sh_file, degree=max_deg, trajectory=trajectory)
    accelerations = gravityModel.load()
    coef = regress(trajectory.positions, accelerations, max_deg, planet)
    #regressor = Regression(trajectory.positions, accelerations, max_degree, planet.radius, planet.mu)

    # Regressor.regress(r, a, deg, planet)
        # Check that regression degree is the same as before, else increase the matrices to be the right size
        # Generate inversion matrix
        # Populate entries in matrix
        # Perform inversion 

if __name__ == "__main__":
    main()