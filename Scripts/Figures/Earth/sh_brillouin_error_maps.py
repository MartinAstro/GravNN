        
import os

import matplotlib.pyplot as plt
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories import DHGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapBase import MapBase


def main():
    # Plotting 
    directory = os.path.abspath('.') +"/Plots/ErrorMaps/"
    os.makedirs(directory, exist_ok=True)

    map_vis = MapBase('mGal')
    map_vis.fig_size = map_vis.half_page
    map_vis.tick_interval = [60, 60]

    my_cmap = 'viridis'
    vlim= [0, 30]

    planet = Earth()
    model_file = planet.sh_file
    density_deg = 180

    df_file = "Data/Dataframes/sh_stats_Brillouin.data"
    trajectory = DHGridDist(planet, planet.radius, degree=density_deg)

    Call_r0_gm = SphericalHarmonics(model_file, degree=1000, trajectory=trajectory)
    Call_a = Call_r0_gm.load().accelerations

    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
    C22_a = C22_r0_gm.load().accelerations

    C00_r0_gm = SphericalHarmonics(model_file, degree=0, trajectory=trajectory)
    C00_a = C00_r0_gm.load().accelerations

    C100_r0_gm = SphericalHarmonics(model_file, degree=100, trajectory=trajectory)
    C100_a = C100_r0_gm.load().accelerations

    C300_r0_gm = SphericalHarmonics(model_file, degree=300, trajectory=trajectory)
    C300_a = C300_r0_gm.load().accelerations

    grid_original = Grid(trajectory=trajectory, accelerations=Call_a-C00_a)
    map_vis.plot_grid(grid_original.total, vlim=None, label=None)#"U_{1000}^{(2)} - U_{100}^{(2)}")
    map_vis.save(plt.gcf(), directory + "sh_brillouin_0.pdf")

    grid_true = Grid(trajectory=trajectory, accelerations=Call_a-C22_a)
    map_vis.plot_grid(grid_true.total, vlim=vlim, label=None)#"U_{1000}^{(2)} - U_{100}^{(2)}")
    map_vis.save(plt.gcf(), directory + "sh_brillouin_2.pdf")

    grid_100 = Grid(trajectory=trajectory, accelerations=C100_a-C22_a)
    diff = grid_true - grid_100
    map_vis.plot_grid(diff.total, vlim=vlim,label=None)#"U_{1000}^{(2)} - U_{100}^{(2)}")
    map_vis.save(plt.gcf(), directory + "sh_brillouin_error_100.pdf")

    grid_300 = Grid(trajectory=trajectory, accelerations=C300_a-C22_a)
    diff = grid_true - grid_300
    map_vis.plot_grid(diff.total, vlim=vlim, label=None)#"U_{1000}^{(2)} - U_{100}^{(2)}")
    map_vis.save(plt.gcf(), directory + "sh_brillouin_error_300.pdf")

    plt.show()
if __name__ == "__main__":
    main()
