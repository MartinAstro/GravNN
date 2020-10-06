from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravityModels.SphericalHarmonics import SphericalHarmonics
from CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.UniformDist import UniformDist
from GravNN.Support.transformations import cart2sph,project_acceleration, check_fix_radial_precision_errors
import os
import pyshtools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def plot_grid_on_map():
    # Plot Grid Points on Perturbations
    planet = Earth()
    radius = planet.radius
    sh_file = planet.sh_hf_file
    # Specify the grid density via the degree
    density_deg = 175
    max_deg = 1000
    trajectory_surf = DHGridDist(planet, radius, degree=density_deg)

    Call_gm = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory_surf)
    Call_grid = Grid(gravityModel=Call_gm)

    C20_gm = SphericalHarmonics(sh_file, 2, trajectory=trajectory_surf)
    C20_grid = Grid(gravityModel=C20_gm)

    map_vis = MapVisualization()
    true_mC20_grid = Call_grid - C20_grid
    point_count = 1000
    size = 1

    #trajectory_uniform = UniformDist(planet, planet.radius, point_count)
    trajectory_random = RandomDist(planet, [planet.radius, planet.radius + 5000], point_count)
    Call_gm = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory_random)
    C20_gm = SphericalHarmonics(sh_file, 2, trajectory=trajectory_random)
    Call_gm.load()
    C20_gm.load()
    acc = Call_gm.accelerations - C20_gm.accelerations

    pos_sphere = cart2sph(trajectory_random.positions)
    pos_sphere = check_fix_radial_precision_errors(pos_sphere)
    acc_proj = project_acceleration(pos_sphere, acc)

    x_train, x_val, y_train, y_val = train_test_split(pos_sphere, acc_proj, random_state=0, test_size=0.001)
    x_train[:,1] =  x_train[:,1]/360*(true_mC20_grid.N_lon-1)
    x_train[:,2] =  x_train[:,2]/180*(true_mC20_grid.N_lat-1)

    fig_pert, ax = map_vis.plot_grid(true_mC20_grid.total, "Acceleration Perturbations")
    ax.scatter( x_train[:,1] ,
                        x_train[:,2],
                        color='r', s=size)

    map_vis.save(plt.gcf(), "RandomDist1000.pdf")
    plt.show()

plot_grid_on_map()