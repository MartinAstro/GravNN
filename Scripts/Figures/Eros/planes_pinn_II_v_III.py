import matplotlib.pyplot as plt
import pandas as pd

from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer


def run(config, model, radius_bounds, max_percent):
    planes_exp = PlanesExperiment(model, config, radius_bounds, 200)
    planes_exp.run()
    vis = PlanesVisualizer(planes_exp)
    vis.fig_size = (vis.w_half, vis.w_half)
    vis.plot(z_max=max_percent)
    return vis


def main():
    """Shows how PINN III avoids extrapolation error
    thanks to design modifications
    """
    df = pd.read_pickle("Data/Dataframes/eros_PINN_III_extrapolation_v3.data")

    # Configuration
    planet = Eros()
    max_radius = planet.radius * 5
    max_percent = 10
    radius_bounds = [-max_radius, max_radius]

    # Load PINN II
    model_id = df["id"].values[-1]
    config, model = load_config_and_model(df, model_id)
    config["obj_file"] = [planet.obj_8k]  # plot with simpler obj file
    vis = run(config, model, radius_bounds, max_percent)
    vis.save(plt.gcf(), "PINNII/Eros_Planes.pdf")

    # PINN III
    model_id = df["id"].values[-2]
    config, model = load_config_and_model(df, model_id)
    config["obj_file"] = [planet.obj_8k]
    vis = run(config, model, radius_bounds, max_percent)
    vis.save(plt.gcf(), "PINNIII/Eros_Planes.pdf")

    plt.show()


if __name__ == "__main__":
    main()
