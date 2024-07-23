import os

import matplotlib
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import sigfig
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        self.fig_size = (self.w_full, self.w_full / 3 * 1.2)

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

    def annotate(self, values):
        avg = sigfig.round(np.nanmean(values), sigfigs=2)
        std = sigfig.round(np.nanstd(values), sigfigs=2)
        max = sigfig.round(np.nanmax(values), sigfigs=2)

        if avg > 1e3:
            stat_str = "%.1E ± %.0E (%.0E)" % (avg, std, max)
        else:
            stat_str = f"{avg}±{std} ({max})"
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
            self.annotate(z_vec)

        if cbar:
            if cbar_gs is None:
                ax = plt.gca()
                divider = make_axes_locatable(ax)
                cbar_gs = divider.append_axes("bottom", size="5%", pad=0.05)
                cBar = plt.colorbar(
                    im,
                    cax=cbar_gs,
                    orientation="horizontal",
                )
            else:
                cBar = plt.colorbar(
                    im,
                    cax=plt.subplot(cbar_gs),
                    orientation="horizontal",
                )
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
        defaults = {"z_min": 1e-3}
        defaults.update(kwargs)
        self.default_plot(x, y, "Acceleration Percent Error", **defaults)

    def plot_RMS(self, **kwargs):
        x = self.experiment.x_test
        y = self.experiment.RMS_acc
        self.default_plot(x, y, "Acceleration RMS Error [$m/s^2$]", **kwargs)

    def plot_gravity_field(self, **kwargs):
        x = self.experiment.x_test
        y = np.linalg.norm(self.experiment.a_test, axis=1, keepdims=True)

        # nan out the interior
        nan_idx = np.where(np.isnan(self.experiment.percent_error_acc))
        y[nan_idx] = np.nan

        label = "Acceleration Magnitude [$m/s^2$]"
        defaults = {"annotate_stats": False}
        defaults.update(kwargs)
        self.default_plot(x, y, colorbar_label=label, cmap=cm.viridis, **defaults)

    def default_plot(self, x, y, colorbar_label, **kwargs):
        fig, ax = self.newFig()

        # If cBar is explicitly set to False, only make one row
        use_cbar = kwargs.get("cbar", True)
        if not use_cbar:
            del fig.axes[0]
            ax0 = plt.subplot(1, 3, 1)
            ax1 = plt.subplot(1, 3, 2)
            ax2 = plt.subplot(1, 3, 3)
            gs = np.array([[ax0, ax1, ax2]])
            # gs = gridspec.GridSpec(1, 3, figure=fig)
            cbar_gs = None
            sca = plt.sca
        else:
            gs = gridspec.GridSpec(
                2,
                3,
                figure=fig,
                width_ratios=[1, 1, 1],
                height_ratios=[1, 0.05],
            )
            cbar_gs = gs[1, :]
            sca = fig.add_subplot

        defaults = {
            "ticks": False,
            "labels": False,
        }
        defaults.update(kwargs)
        defaults.update({"cbar": use_cbar})

        sca(gs[0, 0])
        self.plot_plane(
            x,
            y,
            plane="xy",
            colorbar_label=colorbar_label,
            cbar_gs=cbar_gs,
            **defaults,
        )

        # Only use cbar for first image
        defaults.update({"cbar": False})

        # default to annotate, unless specified
        annotate = kwargs.get("annotate_stats", True)
        defaults.update({"annotate_stats": annotate})
        sca(gs[0, 1])
        self.plot_plane(
            x,
            y,
            plane="xz",
            **defaults,
        )

        # Remove annotate for last image
        sca(gs[0, 2])
        defaults.update({"annotate_stats": False})
        self.plot_plane(
            x,
            y,
            plane="yz",
            **defaults,
        )
