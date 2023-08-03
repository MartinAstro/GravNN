import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Networks.Model import load_config_and_model
from GravNN.Support.Grid import Grid
from GravNN.Trajectories import DHGridDist
from GravNN.Visualization.MapBase import MapBase


class AccMagSortVisualizer:
    def __init__(self, model, config, planet, trajectory):
        self.config = config
        self.model = model
        self.planet = planet
        self.trajectory = trajectory

    def gather_data(self):
        planet = self.planet
        trajectory = self.trajectory
        model = self.model
        config = self.config

        map_vis = MapBase("m/s^2")
        map_vis.fig_size = map_vis.full_page_default
        map_vis.tick_interval = [60, 60]

        # True gravity field
        grav_model = SphericalHarmonics(planet.sh_file, 1000, trajectory).load()
        a_1000 = grav_model.accelerations
        grid_true = Grid(trajectory=trajectory, accelerations=a_1000)

        # Low fidelity estimate
        grav_model = SphericalHarmonics(planet.sh_file, 55, trajectory).load()
        a_sh = grav_model.accelerations
        grid_sh = Grid(trajectory=trajectory, accelerations=a_sh)

        # Planetary oblateness model
        grav_model = SphericalHarmonics(planet.sh_file, 2, trajectory).load()
        a_2 = grav_model.accelerations
        grid_a2 = Grid(trajectory=trajectory, accelerations=a_2)

        # The PINN Model
        a_pinn = model._compute_acceleration(trajectory.positions)
        grid_pinn = Grid(trajectory=trajectory, accelerations=a_pinn)

        # If the PINN model includes point mass + J2, remove it
        if config["deg_removed"][0] == -1:
            grid_pinn -= grid_a2
        grid_true -= grid_a2
        grid_sh -= grid_a2

        self.true_accelerations = grid_true.total.flatten()
        self.sh_accelerations = grid_sh.total.flatten()
        self.pinn_accelerations = grid_pinn.total.flatten()

    def plot(self):
        sorted_idx = np.argsort(self.true_accelerations)

        idx = np.arange(0, len(sorted_idx), 1)
        plt.figure()
        plt.scatter(idx, self.true_accelerations[sorted_idx], label="true", alpha=0.2)
        plt.scatter(idx, self.pinn_accelerations[sorted_idx], label="pinn", alpha=0.2)
        plt.scatter(idx, self.sh_accelerations[sorted_idx], label="sh", alpha=0.2)
        plt.legend()

    def run(self):
        self.gather_data()
        self.plot()


def main():
    # Plotting
    directory = os.path.abspath(".") + "/Plots/"
    os.makedirs(directory, exist_ok=True)

    planet = Earth()
    density_deg = 180
    df_file, idx = "Data/Dataframes/example.data", -1

    df = pd.read_pickle(df_file)
    model_id = df["id"].iloc[idx]
    config, model = load_config_and_model(df, model_id)

    surface_data = DHGridDist(planet, planet.radius, degree=density_deg)

    exp = AccMagSortVisualizer(model, config, planet, surface_data)
    exp.run()
    plt.show()

    # plt.show()


if __name__ == "__main__":
    main()
