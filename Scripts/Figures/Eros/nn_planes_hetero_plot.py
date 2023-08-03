import os

import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Networks.Model import load_config_and_model
from GravNN.Trajectories.PlanesDist import PlanesDist
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer


class PlanesVisualizerModified(PlanesVisualizer):
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)

    def plot(self, percent_max=100, **kwargs):
        from mpl_toolkits.axes_grid1.axes_grid import ImageGrid

        self.max = percent_max
        self.newFig(fig_size=(self.w_full, self.h_quad * 1.25))
        plt.clf()
        x = self.experiment.x_test
        y = self.experiment.losses["percent"] * 100
        cbar_label = "Acceleration Percent Error"
        grid = ImageGrid(
            plt.gcf(),
            111,
            nrows_ncols=(1, 3),
            cbar_mode="single",
            cbar_location="bottom",
        )
        plt.sca(grid[0])
        self.plot_plane(
            x,
            y,
            plane="xy",
            colorbar_label=cbar_label,
            cbar=False,
            ticks=False,
            labels=False,
            **kwargs,
        )
        plt.sca(grid[1])
        self.plot_plane(
            x,
            y,
            plane="xz",
            colorbar_label=cbar_label,
            cbar=False,
            ticks=False,
            labels=False,
            **kwargs,
        )
        plt.sca(grid[2])
        im = self.plot_plane(
            x,
            y,
            plane="yz",
            colorbar_label=cbar_label,
            cbar=False,
            ticks=False,
            labels=False,
            **kwargs,
        )
        cbar = grid.cbar_axes[0].colorbar(im)
        cbar.ax.set_xlabel(cbar_label)
        plt.tight_layout()


class PlanesExperimentPolyhedral(PlanesExperiment):
    def __init__(
        self,
        model,
        config,
        radius_bounds,
        samples_1d,
        gravity_data_fcn,
        **kwargs,
    ):
        super().__init__(model, config, radius_bounds, samples_1d)
        self.gravity_data_fcn = gravity_data_fcn

    def get_model_data(self):
        planet = self.config["planet"][0]
        obj_file = self.config.get("grav_file", [None])[0]
        gravity_data_fcn = self.gravity_data_fcn
        interpolation_dist = PlanesDist(
            planet,
            bounds=self.bounds,
            samples_1d=self.samples_1d,
            **self.config,
        )

        full_dist = interpolation_dist

        x, a, u = gravity_data_fcn(full_dist, obj_file, **self.config)

        self.x_pred = x
        self.a_pred = a
        self.u_pred = u


def main():
    df = pd.read_pickle("Data/Dataframes/heterogenous_eros_041823.data")
    model_id = df["id"].values[-1]
    print(model_id)

    config, model = load_config_and_model(df, model_id)

    planet = config["planet"][0]
    radius_bounds = [-planet.radius * 3, planet.radius * 3]
    max_percent = 10

    planes_exp = PlanesExperiment(model, config, radius_bounds, 200)
    planes_exp.run()
    vis_hetero = PlanesVisualizerModified(
        planes_exp,
        save_directory=os.path.abspath(".") + "/Plots/Eros/",
    )
    vis_hetero.plot(percent_max=max_percent, annotate_stats=True)
    vis_hetero.save(plt.gcf(), "Eros_Planes_hetero.pdf")

    # Plot the polyhedral model
    planes_exp = PlanesExperimentPolyhedral(
        model,
        config,
        radius_bounds,
        200,
        get_poly_data,
    )
    planes_exp.run()
    vis = PlanesVisualizerModified(
        planes_exp,
        save_directory=os.path.abspath(".") + "/Plots/Eros/",
    )
    vis.interior_mask = vis_hetero.interior_mask  # speeds up calc
    vis.plot(percent_max=max_percent, annotate_stats=True)
    vis.save(plt.gcf(), "Eros_Planes_homo.pdf")

    # plt.show()


if __name__ == "__main__":
    main()
