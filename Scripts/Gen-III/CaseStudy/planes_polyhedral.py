import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.GravityModels.HeterogeneousPoly import get_hetero_poly_data
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer


class PlanesVisualizerContour(PlanesVisualizer):
    def __init__(self, experiment, **kwargs):
        super().__init__(experiment, **kwargs)
        self.fig_size = (self.w_half, self.w_half)

    def plot_plane(
        self,
        x_vec,
        z_vec,
        plane="xy",
        colorbar_label=None,
        srp_sphere=False,
        annotate_stats=False,
        labels=True,
        ticks=True,
        cbar=True,
        cmap=cm.jet,
        cbar_gs=None,
        z_min=None,
        z_max=None,
        log=False,
        contour=False,
        trajectory=None,
        **kwargs,
    ):
        # create mask for the two position coordinates
        mask = self.plane_mask(plane)

        # select the indices for the plane
        idx_start, idx_end = self.get_plane_idx(plane)

        # Grab the positions for the plane and remove the irrelevant coordinate
        x = x_vec[idx_start:idx_end, mask]

        # Select the metric of interest
        z = z_vec[idx_start:idx_end]
        z_min_true = np.nanmin(z)
        z_max_true = np.nanmax(z)

        if z_min is None:
            z_min = z_min_true
        if z_max is None:
            z_max = z_max_true

        # normalize position coordinates w.r.t. radius
        min_x_0 = np.min(x[:, 0]) / self.radius
        max_x_0 = np.max(x[:, 0]) / self.radius

        min_x_1 = np.min(x[:, 1]) / self.radius
        max_x_1 = np.max(x[:, 1]) / self.radius

        N = np.sqrt(len(z)).astype(int)

        if log:
            norm = matplotlib.colors.LogNorm(vmin=z_min, vmax=z_max)
        else:
            norm = matplotlib.colors.Normalize(vmin=z_min, vmax=z_max)

        im = plt.imshow(
            z.reshape((N, N)),
            extent=[min_x_0, max_x_0, min_x_1, max_x_1],
            origin="lower",
            cmap=cmap,
            norm=norm,
        )

        # optional contour
        if contour:
            zm = np.ma.masked_invalid(z)
            cntr = plt.gca().contour(
                zm.reshape((N, N)),
                levels=[5, 10, 20, 40, 80],  # np.logspace(z_min, z_max, 5),
                norm=norm,
                extent=[min_x_0, max_x_0, min_x_1, max_x_1],
                colors="black",
                linewidths=0.5,
            )

            plt.gca().clabel(cntr, inline=True, fontsize=8)

        # overlay a trajectory
        if trajectory is not None:
            X0, X1 = trajectory[:, mask] / self.radius
            plt.plot(X0, X1, color="black", linewidth=0.5)

        plt.gca().set_xlabel(plane[0])
        plt.gca().set_ylabel(plane[1])

        if not labels:
            plt.gca().set_xlabel("")
            plt.gca().set_ylabel("")

        if not ticks:
            plt.gca().set_xticks([])
            plt.gca().set_yticks([])
            plt.gca().set_xticklabels("")
            plt.gca().set_yticklabels("")

        if annotate_stats:
            self.annotate(z_vec)

        if cbar:
            if cbar_gs is None:
                ax = plt.gca()
                divider = make_axes_locatable(ax)
                cbar_gs = divider.append_axes("right", size="5%", pad=0.05)
                cBar = plt.colorbar(
                    im,
                    cax=cbar_gs,
                    orientation="vertical",
                )
            else:
                cBar = plt.colorbar(
                    im,
                    cax=plt.subplot(cbar_gs),
                    orientation="horizontal",
                )
            cBar.set_label(colorbar_label)

        return im

    def plot(self, **kwargs):
        x = self.experiment.x_test
        y = self.experiment.percent_error_acc
        defaults = {"z_min": 1e-3}
        defaults.update(**kwargs)

        fig, ax = self.newFig()
        use_cbar = kwargs.get("cbar", True)

        defaults = {
            "ticks": False,
            "labels": False,
        }
        defaults.update(kwargs)
        defaults.update({"cbar": use_cbar})

        self.plot_plane(
            x,
            y,
            plane="xy",
            colorbar_label="Acceleration Percent Error",
            contour=True,
            **defaults,
        )


def main():
    df = pd.read_pickle("Data/Dataframes/pinn_primary_figure_III.data")
    model_id = df["id"].values[-1]
    print(model_id)

    config, model = load_config_and_model(df, model_id)
    config["gravity_data_fcn"] = [get_hetero_poly_data]

    planet = config["planet"][0]
    radius_bounds = [-planet.radius * 3, planet.radius * 3]
    max_percent = 10

    model = Polyhedral(planet, planet.obj_200k)

    planes_exp = PlanesExperiment(
        model,
        config,
        radius_bounds,
        200,
    )
    planes_exp.load_model_data(model)
    planes_exp.run()
    vis_hetero = PlanesVisualizerContour(planes_exp)
    vis_hetero.plot(z_max=max_percent)
    vis_hetero.save(plt.gcf(), "Eros_Planes_homo")
    plt.show()


if __name__ == "__main__":
    main()
