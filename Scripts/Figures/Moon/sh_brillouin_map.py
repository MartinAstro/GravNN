import os

import matplotlib.pyplot as plt

from GravNN.CelestialBodies.Planets import Moon
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Support.Grid import Grid
from GravNN.Trajectories import DHGridDist
from GravNN.Visualization.MapBase import MapBase


def main():
    planet = Moon()
    sh_file = planet.sh_file
    density_deg = 180
    max_deg = 1000

    radius_min = planet.radius

    trajectory = DHGridDist(planet, radius_min, degree=density_deg)

    Call_r0_gm = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
    Call_a = Call_r0_gm.load().accelerations

    C55_r0_gm = SphericalHarmonics(sh_file, degree=55, trajectory=trajectory)
    C55_a = C55_r0_gm.load().accelerations

    C22_r0_gm = SphericalHarmonics(sh_file, degree=2, trajectory=trajectory)
    C22_a = C22_r0_gm.load().accelerations

    grid_true = Grid(trajectory=trajectory, accelerations=Call_a - C22_a)
    grid_55 = Grid(trajectory=trajectory, accelerations=C55_a - C22_a)

    directory = os.path.abspath(".") + "/Plots/Moon/"
    os.makedirs(directory, exist_ok=True)

    mapUnit = "mGal"
    map_vis = MapBase(mapUnit)
    map_vis.fig_size = map_vis.full_page

    my_cmap = "viridis"
    vlim = [0, 60]
    im = map_vis.new_map(grid_true.total, vlim=vlim, cmap=my_cmap)
    map_vis.add_colorbar(im, "[mGal]", vlim=vlim, extend="max")
    map_vis.save(plt.gcf(), directory + "sh_brillouin_map.pdf")

    map_vis.newFig()
    im = map_vis.new_map(grid_55.total, vlim=vlim, cmap=my_cmap)
    map_vis.add_colorbar(im, "[mGal]", vlim=vlim, extend="max", pad=0.05)
    map_vis.save(plt.gcf(), directory + "sh_brillouin_55_map.pdf")

    mapUnit = "mGal"
    map_vis.fig_size = map_vis.half_page
    map_vis.tick_interval = [60, 60]

    map_vis.newFig()
    my_cmap = "viridis"
    vlim = [0, 40]
    im = map_vis.new_map(grid_true.total, vlim=vlim, cmap=my_cmap)
    map_vis.add_colorbar(
        im,
        "[mGal]",
        vlim=vlim,
        extend="max",
        loc="top",
        orientation="horizontal",
        pad=0.05,
    )
    map_vis.save(plt.gcf(), directory + "sh_brillouin_map_half.pdf")

    plt.show()


if __name__ == "__main__":
    main()
