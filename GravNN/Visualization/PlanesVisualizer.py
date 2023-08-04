import os

import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import sigfig
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import gaussian_kde

from GravNN.Visualization.VisualizationBase import VisualizationBase


class PlanesVisualizer(VisualizationBase):
    def __init__(self, experiment, **kwargs):
        super().__init__(**kwargs)
        self.experiment = experiment
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.radius = self.experiment.config["planet"][0].radius
        self.training_bounds = np.array(self.experiment.training_bounds)
        self.planet = experiment.config["planet"][0]
        self.interior_mask = self.experiment.interior_mask

    def plane_mask(self, plane):
        mask = np.array([False, False, False])
        if "x" in plane:
            mask = np.array([True, False, False]) + mask
        if "y" in plane:
            mask = np.array([False, True, False]) + mask
        if "z" in plane:
            mask = np.array([False, False, True]) + mask
        mask = mask.astype(bool)
        return mask

    def get_plane_idx(self, plane):
        N = len(self.experiment.x_test)
        M = N // 3
        if plane == "xy":
            idx_start = 0
            idx_end = M
        elif plane == "xz":
            idx_start = M
            idx_end = 2 * M
        elif plane == "yz":
            idx_start = 2 * M
            idx_end = 3 * M
        else:
            raise ValueError(f"Invalid Plane: {plane}")
        return idx_start, idx_end

    def set_SRP_contour(self, a_srp):
        self.r_srp = np.sqrt(self.planet.mu / a_srp)

    def plot_density_map(self, x_vec, plane="xy"):
        x, y, z = x_vec.T
        mask = self.plane_mask(plane)
        train_data = x_vec.T[mask]
        test_data = self.experiment.x_test.T[mask]
        k = gaussian_kde(train_data)
        nbins = 50

        x_0, x_1 = train_data[0], train_data[1]
        x_test_0, x_test_1 = test_data[0], test_data[1]

        x_0_i, x_1_i = np.mgrid[
            x_0.min() : x_0.max() : nbins * 1j,
            x_1.min() : x_1.max() : nbins * 1j,
        ]
        k(np.vstack([x_0_i.flatten(), x_1_i.flatten()]))
        # plt.gca().pcolormesh(x_0_i, x_1_i, zi.reshape(x_0_i.shape))

        min_x_test_0 = np.min(x_test_0) / self.radius
        max_x_test_0 = np.max(x_test_0) / self.radius

        min_x_test_1 = np.min(x_test_1) / self.radius
        max_x_test_1 = np.max(x_test_1) / self.radius

        np.min(x_0) / self.radius
        np.max(x_0) / self.radius

        np.min(x_1) / self.radius
        np.max(x_1) / self.radius

        heatmap, xedges, yedges = np.histogram2d(
            x_0 / self.radius,
            x_1 / self.radius,
            nbins,
            # range=[
            #     [min_x_0, max_x_0],
            #     [min_x_1, max_x_1]
            #     ]
            range=[
                [min_x_test_0, max_x_test_0],
                [min_x_test_1, max_x_test_1],
            ],
        )

        heatmap = gaussian_filter(heatmap, sigma=1)

        extent = [min_x_test_0, max_x_test_0, min_x_test_1, max_x_test_1]
        # extent = [min_x_0, max_x_0, min_x_1, max_x_1]
        plt.imshow(heatmap.T, extent=extent, origin="lower", cmap=cm.jet)
        cbar = plt.colorbar(fraction=0.15)
        cbar.set_label("Samples")
        # plt.scatter(x_0 / self.radius,
        #             x_1 / self.radius,
        #              s=5, c='r', alpha=0.5)
        # plt.scatter(x_0, x_1, c='r', alpha=1)
        # plt.xlim([min_x_test_0, max_x_test_0])
        # plt.ylim([min_x_test_1, max_x_test_1])

        plt.xlabel(plane[0])
        plt.ylabel(plane[1])

    def annotate(self, values):
        avg = sigfig.round(np.nanmean(values), sigfigs=2)
        std = sigfig.round(np.nanstd(values), sigfigs=2)
        max = sigfig.round(np.nanmax(values), sigfigs=2)

        if avg > 1e3:
            stat_str = "%.1E ± %.1E (%.1E)" % (avg, std, max)
        else:
            stat_str = f"{avg}±{std} ({max})"
        plt.sca(plt.gcf().axes[1])
        plt.gca().annotate(
            stat_str,
            xy=(0.5, 0.1),
            ha="center",
            va="center",
            xycoords="axes fraction",
            bbox=dict(boxstyle="round", fc="w"),
        )

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
        z_min=0,
        z_max=None,
        log=False,
        contour=False,
        trajectory=False,
    ):
        # create mask for the two position coordinates
        mask = self.plane_mask(plane)

        # select the indices for the plane
        idx_start, idx_end = self.get_plane_idx(plane)

        # Grab the positions for the plane and remove the irrelevant coordinate
        x = x_vec[idx_start:idx_end, mask]

        # Select the metric of interest
        z = z_vec[idx_start:idx_end]

        # normalize position coordinates w.r.t. radius
        min_x_0 = np.min(x[:, 0]) / self.radius
        max_x_0 = np.max(x[:, 0]) / self.radius

        min_x_1 = np.min(x[:, 1]) / self.radius
        max_x_1 = np.max(x[:, 1]) / self.radius

        N = np.sqrt(len(z)).astype(int)

        if log:
            if z_min == 0:
                print("WARNING: Log scale cannot work with z_min = 0, setting to 1e-3")
                z_min = 1e-3
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
                levels=np.logspace(z_min, z_max, 5),
                norm=norm,
                extent=[min_x_0, max_x_0, min_x_1, max_x_1],
                colors="k",
                linewidths=0.5,
            )

            plt.clabel(cntr, inline=True, fontsize=8, fmt="%1.0e")

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
            self.annotate(z)

        if cbar:
            if cbar_gs is None:
                ax = plt.gca()
                divider = make_axes_locatable(ax)
                cbar_gs = divider.append_axes("right", size="5%", pad=0.05)
                cBar = plt.colorbar(im, cax=cbar_gs)
            else:
                cBar = plt.colorbar(im, cax=plt.subplot(cbar_gs))
            cBar.set_label(colorbar_label)

        if srp_sphere:
            circ = matplotlib.patches.Circle(
                (0, 0),
                self.r_srp / self.radius,
                fill=False,
                edgecolor="white",
            )
            plt.gca().add_patch(circ)

        return im

    def plot(self, **kwargs):
        # default plotting routine
        self.plot_percent_error(**kwargs)

    def plot_percent_error(self, **kwargs):
        x = self.experiment.x_test
        y = self.experiment.percent_error_acc
        self.default_plot(x, y, "Acceleration Percent Error", **kwargs)

    def plot_RMS(self, **kwargs):
        x = self.experiment.x_test
        y = self.experiment.RMS_acc
        self.default_plot(x, y, "Acceleration RMS Error [$m/s^2$]", **kwargs)

    def plot_gravity_field(self, **kwargs):
        x = self.experiment.x_test
        y = np.linalg.norm(self.experiment.a_test, axis=1, keepdims=True)
        label = "Acceleration Magnitude [$m/s^2$]"
        self.default_plot(x, y, colorbar_label=label, cmap=cm.viridis, **kwargs)

    def default_plot(self, x, y, colorbar_label, **kwargs):
        fig, ax = self.newFig()
        x = self.experiment.x_test
        y = self.experiment.percent_error_acc
        gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

        kwargs.update({"ticks": False, "labels": False})

        fig.add_subplot(gs[0, 0])
        self.plot_plane(x, y, plane="xy", cbar=False, **kwargs)

        fig.add_subplot(gs[0, 1])
        self.plot_plane(x, y, plane="xz", cbar=False, **kwargs)

        fig.add_subplot(gs[0, 2])
        self.plot_plane(
            x,
            y,
            plane="yz",
            cbar=True,
            colorbar_label=colorbar_label,
            cbar_gs=gs[3],
            **kwargs,
        )
        plt.subplots_adjust(wspace=0.00, hspace=0.00)
