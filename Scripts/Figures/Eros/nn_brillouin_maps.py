import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories import DHGridDist
from GravNN.Support.Grid import Grid
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.FigureSupport import format_potential_as_Nx3, clamp
from sigfig import round

def main():
    directory = os.path.abspath(".") + "/Plots/Asteroid/Eros/"
    os.makedirs(directory, exist_ok=True)

    map_vis = MapVisualization(halt_formatting=False)
    map_vis.fig_size = map_vis.half_page
    map_vis.tick_interval = [60, 60]

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

    #df, descriptor = pd.read_pickle("Data/Dataframes/useless_070721_v1.data"), "[0,10000]"
    #df, descriptor = pd.read_pickle("Data/Dataframes/useless_070721_v2.data"), "[5000,10000]"
    df, descriptor = pd.read_pickle("Data/Dataframes/useless_070621_v4.data"), "[5000,10000]*2500 + [0, 5000]*250"
    model_ids = df["id"].values[:]
    for model_id in model_ids:
        config, model = load_config_and_model(model_id, df)
        extra_samples = config.get('extra_N_train', [None])[0]
        directory = (
            os.path.abspath(".")
            + "/Plots/Asteroid/"
            + config["PINN_constraint_fcn"][0].__name__
            + "/"
            + str(np.round(config["radius_min"][0], 2))
            + "_"
            + str(np.round(config["radius_max"][0], 2))
            + "_"
            + str(extra_samples)
            + "/"
        )
        os.makedirs(directory, exist_ok=True)

        # Brillouin Sphere
        x = poly_gm.positions
        data_pred = model.generate_nn_data(x)
        grid_pot_pred = Grid(
            trajectory=DH_trajectory,
            accelerations=format_potential_as_Nx3(data_pred["u"]),
            transform=False,
        )
        grid_acc_pred = Grid(trajectory=DH_trajectory, accelerations=data_pred["a"])

        # Network produced accelerations and differences
        map_vis.plot_grid(grid_acc_pred.total, '$m/s^2$', orientation='horizontal',loc='top', labels=False, ticks=False)
        map_vis.save(plt.gcf(), directory + "nn_brill_acc.pdf")

        grid_acc_diff = grid_acc_true - grid_acc_pred
        avg = round(float(np.average(grid_acc_diff.total)),sigfigs=2)
        std = 3*np.std(grid_acc_diff.total)
        map_vis.plot_grid(grid_acc_diff.total, '$m/s^2$', orientation='horizontal',loc='top', labels=False, ticks=False, vlim=[clamp(avg - std, 0, np.inf), avg + std])
        plt.gcf().axes[0].annotate("Avg: " + str(avg), xy=(0.05,0.05), xycoords='axes fraction', fontsize='small', c='white')
        map_vis.save(plt.gcf(), directory + "nn_brill_acc_diff.pdf")

        # Network produced potentials and differences
        map_vis.plot_grid(grid_pot_pred.total, '$m^2/s^2$', orientation='horizontal',loc='top', labels=False, ticks=False)
        map_vis.save(plt.gcf(), directory + "nn_brill_pot.pdf")

        grid_pot_diff = grid_pot_true - grid_pot_pred
        avg = round(float(np.average(grid_pot_diff.total)),sigfigs=2)
        std = 3*np.std(grid_pot_diff.total)
        map_vis.plot_grid(grid_pot_diff.total, '$m^2/s^2$', orientation='horizontal',loc='top', labels=False, ticks=False, vlim=[clamp(avg - std, 0, np.inf), avg + std])
        plt.gcf().axes[0].annotate("Avg: " + str(avg), xy=(0.05,0.05), xycoords='axes fraction', fontsize='small', c='white')
        map_vis.save(plt.gcf(), directory + "nn_brill_pot_diff.pdf")


    #plt.show()


if __name__ == "__main__":
    main()
