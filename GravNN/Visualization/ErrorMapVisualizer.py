import matplotlib.pyplot as plt
import numpy as np

from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapBase import MapBase


class ErrorMapVisualizer(MapBase):
    def __init__(self, config, model, sh_deg):
        super().__init__("m/s^2")
        self.config = config
        self.model = model
        self.sh_deg = sh_deg
        self.fig_size = self.half_page_default
        self.tick_interval = [60, 60]

    def plot_models(self, grid_true, grid_pinn, grid_sh):
        plt.figure(figsize=(8, 12))

        plt.subplot(3, 1, 1)
        vlim = [
            grid_pinn.total.min(),
            np.mean(grid_pinn.total) + 2 * np.std(grid_pinn.total),
        ]
        self.plot_grid(grid_true.total, vlim=vlim, label=None, new_fig=False)
        plt.gcf().get_axes()[-2].set_title("True Model")

        plt.subplot(3, 1, 2)
        self.plot_grid(grid_pinn.total, vlim=vlim, label=None, new_fig=False)
        plt.gcf().get_axes()[-2].set_title("PINN Model")

        plt.subplot(3, 1, 3)
        self.plot_grid(grid_sh.total, vlim=vlim, label=None, new_fig=False)
        plt.gcf().get_axes()[-2].set_title("SH Model")

    def plot_percent_error(self, sh_percent, pinn_percent):
        vlim = [0, 100]
        plt.figure(figsize=(8, 12))

        plt.subplot(2, 1, 1)
        self.plot_grid(pinn_percent.total, vlim=vlim, label=None, new_fig=False)
        plt.gcf().get_axes()[-2].set_title(
            f"PINN Percent Error: {np.average(pinn_percent.total)}",
        )

        plt.subplot(2, 1, 2)
        self.plot_grid(sh_percent.total, vlim=vlim, label=None, new_fig=False)
        plt.gcf().get_axes()[-2].set_title(
            f"SH Percent Error: {np.average(sh_percent.total)}",
        )

    def plot_rms(self, sh_diff, pinn_diff):
        vlim = [
            pinn_diff.total.min(),
            np.mean(pinn_diff.total) + 2 * np.std(pinn_diff.total),
        ]
        plt.figure(figsize=(8, 12))

        plt.subplot(2, 1, 1)
        self.plot_grid(pinn_diff.total, vlim=vlim, label=None, new_fig=False)
        plt.gcf().get_axes()[-2].set_title(
            f"PINN RMS Diff: {np.average(pinn_diff.total)}",
        )

        plt.subplot(2, 1, 2)
        self.plot_grid(sh_diff.total, vlim=vlim, label=None, new_fig=False)
        plt.gcf().get_axes()[-2].set_title(f"SH RMS Diff: {np.average(sh_diff.total)}")

    def plot(self, trajectory):
        planet = self.config["planet"][0]

        # True Model
        grav_model = SphericalHarmonics(planet.sh_file, 1000, trajectory).load()
        a_1000 = grav_model.accelerations
        grid_true = Grid(trajectory=trajectory, accelerations=a_1000)

        # Low fidelity SH Model
        grav_model = SphericalHarmonics(planet.sh_file, self.sh_deg, trajectory).load()
        a_sh = grav_model.accelerations
        grid_sh = Grid(trajectory=trajectory, accelerations=a_sh)

        # Planetary oblateness model
        grav_model = SphericalHarmonics(planet.sh_file, 2, trajectory).load()
        a_2 = grav_model.accelerations
        grid_a2 = Grid(trajectory=trajectory, accelerations=a_2)

        # The PINN Model
        a_pinn = self.model._compute_acceleration(trajectory.positions)
        grid_pinn = Grid(trajectory=trajectory, accelerations=a_pinn)

        # If the PINN model includes point mass + J2, remove it
        if self.config["deg_removed"][0] == -1:
            grid_pinn -= grid_a2
        grid_true -= grid_a2
        grid_sh -= grid_a2

        pinn_diff = grid_pinn - grid_true
        sh_diff = grid_sh - grid_true

        pinn_percent = pinn_diff / grid_pinn * 100
        sh_percent = sh_diff / grid_sh * 100

        self.plot_models(grid_true, grid_pinn, grid_sh)
        self.plot_percent_error(sh_percent, pinn_percent)
        self.plot_rms(sh_diff, pinn_diff)
