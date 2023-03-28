import os

import matplotlib.pyplot as plt
import pandas as pd

from GravNN.CelestialBodies.Planets import Earth
from GravNN.Networks.Model import load_config_and_model
from GravNN.Trajectories import DHGridDist
from GravNN.Visualization.ErrorMapVisualizer import ErrorMapVisualizer


def main():
    # Plotting
    directory = os.path.abspath(".") + "/Plots/"
    os.makedirs(directory, exist_ok=True)

    planet = Earth()
    density_deg = 180
    # density_deg = 80 # 50000
    # density_deg = 25 # 5000
    df_file, idx = "Data/Dataframes/earth_revisited_032723.data", -1

    df = pd.read_pickle(df_file)
    model_id = df["id"].iloc[idx]
    config, model = load_config_and_model(model_id, df)

    # plt.switch_backend("WebAgg")
    vis = ErrorMapVisualizer(config, model, sh_deg=55)

    surface_data = DHGridDist(planet, planet.radius, degree=density_deg)
    vis.plot(surface_data)

    low_altitude_data = DHGridDist(planet, planet.radius + 100000, degree=density_deg)
    vis.plot(low_altitude_data)

    LEO_data = DHGridDist(planet, planet.radius + 420000, degree=density_deg)
    vis.plot(LEO_data)

    # high_alt_data = DHGridDist(planet, planet.radius*10, degree=density_deg)
    # plot(config, model, high_alt_data, sh_deg)

    plt.show()


if __name__ == "__main__":
    main()
