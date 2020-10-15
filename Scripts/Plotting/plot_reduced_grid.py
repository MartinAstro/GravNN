from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist

import os
import pyshtools
import matplotlib.pyplot as plt
import numpy as np
import pickle

mapUnit = "m/s^2"
mapUnit = 'mGal'
map_vis = MapVisualization(unit=mapUnit)

planet = Earth()
radius = planet.radius
model_file = planet.sh_hf_file
density_deg = 175
max_deg = 1000

trajectory_reduced = ReducedGridDist(planet, radius, degree=density_deg, reduction=0.25)
Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_reduced)
Call_r0_grid = Grid(trajectory=trajectory_reduced, accelerations=Call_r0_gm.load())
C20_r0_gm= SphericalHarmonics(model_file, degree=2, trajectory=trajectory_reduced)
C20_r0_grid = Grid(trajectory=trajectory_reduced, accelerations=C20_r0_gm.load())
R0_pert_grid = Call_r0_grid - C20_r0_grid

def plot_sh_model():
    fig_pert, ax = map_vis.plot_grid(R0_pert_grid.total, "Acceleration [mGal]")
    plt.show()

def main():
    plot_sh_model()


if __name__ == "__main__":
    main()
