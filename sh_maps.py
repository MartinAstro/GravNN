from Visualization.Grid import Grid
from Visualization.MapVisualization import MapVisualization
from GravityModels.SphericalHarmonics import SphericalHarmonics
from CelestialBodies.Planets import Earth
from Trajectories.DHGridDist import DHGridDist
from GravityModels.NN_Base import NN_Base
import os
import pyshtools
import matplotlib.pyplot as plt
import numpy as np
from Visualization.MapVizFunctions import * 
def main():

    planet = Earth()
    radius = planet.radius+50000.0
    # Specify the grid density via the degree
    trajectory = DHGridDist(planet, radius, degree=175)
    
    # Main 
    map_viz = MapVisualization()

    Call_gm = SphericalHarmonics(planet.sh_file, degree=None, trajectory=trajectory)
    Call_grid = Grid(gravityModel=Call_gm)

    C20_gm = SphericalHarmonics(planet.sh_file, 2, trajectory=trajectory)
    C20_grid = Grid(gravityModel=C20_gm)

    Clm_gm = SphericalHarmonics(planet.sh_file, 100, trajectory=trajectory)
    Clm_grid = Grid(gravityModel=Clm_gm)

    high_fidelity_maps(Call_grid, C20_grid)
    percent_error_maps(Call_grid, Clm_grid, C20_grid, vlim=500)
    # component_error([5, 10, 20, 50, 100, 150], radius, planet.sh_file, Call_grid, C20_grid) # Takes a long time

    plt.show()




if __name__ == "__main__":
    main()
