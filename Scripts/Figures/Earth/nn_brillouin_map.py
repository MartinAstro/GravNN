import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories import DHGridDist
from GravNN.Visualization.ErrorMapVisualizer import ErrorMapVisualizer
from GravNN.Networks.Model import load_config_and_model

def main():
    # Plotting 
    directory = os.path.abspath('.') +"/Plots/"
    os.makedirs(directory, exist_ok=True)

    planet = Earth()
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
    
<<<<<<< HEAD
    # LEO_data = DHGridDist(planet, planet.radius+420000, degree=density_deg)
    # plot(df, idx, planet, LEO_data)
    
    # high_alt_data = DHGridDist(planet, planet.radius*10, degree=density_deg)
    # plot(df, idx, planet, high_alt_data)

=======
    LEO_data = DHGridDist(planet, planet.radius+420000, degree=density_deg)
    vis.plot(LEO_data)
    
    # high_alt_data = DHGridDist(planet, planet.radius*10, degree=density_deg)
    # plot(config, model, high_alt_data, sh_deg)
>>>>>>> a53738c (change map visualization to class)

    plt.show()
    # plt.show()
if __name__ == "__main__":
    main()
