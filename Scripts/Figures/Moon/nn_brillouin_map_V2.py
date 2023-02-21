        
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Moon
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories import DHGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.ErrorMapVisualizer import ErrorMapVisualizer
from GravNN.Networks.Model import load_config_and_model


def main():
    planet = Moon()
    density_deg = 180
    # density_deg = 80 # 50000
    # density_deg = 25 # 5000
    df_file, idx = "Data/Dataframes/example.data", -1

    df = pd.read_pickle(df_file)
    model_id = df['id'].iloc[idx]
    config, model = load_config_and_model(model_id, df)

    # plt.switch_backend("WebAgg")
    vis = ErrorMapVisualizer(config, model, sh_deg=55)

    surface_data = DHGridDist(planet, planet.radius, degree=density_deg)
    vis.plot(surface_data)
    
    LEO_data = DHGridDist(planet, planet.radius+50000, degree=density_deg)
    vis.plot(LEO_data)

    plt.show()
if __name__ == "__main__":
    main()
