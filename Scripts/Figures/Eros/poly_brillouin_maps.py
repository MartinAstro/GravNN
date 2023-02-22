import os
import matplotlib.pyplot as plt
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories import DHGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapBase import MapBase
from GravNN.Visualization.FigureSupport import format_potential_as_Nx3


def main():
    directory = os.path.abspath(".") + "/Plots/Asteroid/Eros/"
    os.makedirs(directory, exist_ok=True)

    map_vis = MapBase()
    map_vis.fig_size = map_vis.full_page

    planet = Eros()
    density_deg = 180

    radius_min = planet.radius
    DH_trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    poly_gm = Polyhedral(planet, planet.obj_8k, trajectory=DH_trajectory).load(
        override=False
    )

    u_3vec = format_potential_as_Nx3(poly_gm.potentials)
    grid_pot_true = Grid(
        trajectory=DH_trajectory, accelerations=u_3vec, transform=False
    )
    grid_acc_true = Grid(trajectory=DH_trajectory, accelerations=poly_gm.accelerations)

    map_vis.plot_grid(grid_acc_true.total, label="$m/s^2$")
    map_vis.save(plt.gcf(), directory + "poly_brill_acc.pdf")

    map_vis.plot_grid(grid_pot_true.total, label="$m^2/s^2$")
    map_vis.save(plt.gcf(), directory + "poly_brill_pot.pdf")

    plt.show()


if __name__ == "__main__":
    main()
