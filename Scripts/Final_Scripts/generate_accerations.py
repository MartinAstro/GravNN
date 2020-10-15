from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.UniformDist import UniformDist
from GravNN.Support.transformations import cart2sph
import os
import pyshtools
import matplotlib.pyplot as plt
import numpy as np
import pickle

mapUnit = "m/s^2"
mapUnit = 'mGal'
map_vis = MapVisualization(unit=mapUnit)
#map_vis.fig_size = (3, 1.8)

# Plot Grid Points on Perturbations
planet = Earth()
radius = planet.radius
model_file = planet.sh_hf_file
# Specify the grid density via the degree
density_deg = 175
max_deg = 1000
trajectory_surf = DHGridDist(planet, radius, degree=density_deg)
Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory_surf)
Call_r0_grid = Grid(trajectory=trajectory_surf, accelerations=Call_r0_gm.load())
C20_r0_gm= SphericalHarmonics(model_file, degree=2, trajectory=trajectory_surf)
C20_r0_grid = Grid(trajectory=trajectory_surf, accelerations=C20_r0_gm.load())
R0_pert_grid = Call_r0_grid - C20_r0_grid