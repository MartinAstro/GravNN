        
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories import DHGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Networks.Model import load_config_and_model


def main():
    # Plotting 
    directory = os.path.abspath('.') +"/Plots/"
    os.makedirs(directory, exist_ok=True)

    map_vis = MapVisualization('mGal')
    map_vis.fig_size = map_vis.full_page
    #map_vis.tick_interval = [60, 60]

    my_cmap = 'viridis'
    # vlim= [0, 30]
    vlim = None

    planet = Earth()
    model_file = planet.sh_file
    density_deg = 180

    df_file ='C:\\Users\\John\\Documents\\Research\\ML_Gravity\\Data\\Dataframes\\pinn_df.data'
    df_file ='Data/Dataframes/earth_trajectory_v2.data'
    df_file ='Data/Dataframes/useless_072621_v2.data'
    df_file ='Data/Dataframes/earth_pinn_III_v2.data'

    df = pd.read_pickle(df_file)

    trajectory = DHGridDist(planet, planet.radius, degree=density_deg)
    # trajectory = DHGridDist(planet, planet.radius+420000, degree=density_deg)
    grav_model = SphericalHarmonics(planet.sh_file,1000,trajectory).load()
    a_1000 = grav_model.accelerations
    grid_1000 = Grid(trajectory=trajectory, accelerations=a_1000)

    grav_model = SphericalHarmonics(planet.sh_file,2,trajectory).load()
    a_2 = grav_model.accelerations
    grid_true = grid_1000 - Grid(trajectory=trajectory, accelerations=a_2)

    # for i in range(len(df)):
    # with -1 as i:
    i = -1
    row = df.iloc[i]
    model_id = row['id']
    config, model = load_config_and_model(model_id, df)

    a_pred = model.generate_acceleration(trajectory.positions)
    grid_pred = Grid(trajectory=trajectory, accelerations=a_pred)

    # The learned PINN model 
    map_vis.plot_grid(grid_pred.total, vlim=vlim, label=None)#"U_{1000}^{(2)} - U_{100}^{(2)}")
    #map_vis.save(plt.gcf(), directory + "pinn_brillouin_" + str(row['num_units']) + ".pdf")

    # The True Model
    map_vis.plot_grid(grid_true.total, vlim=vlim, label=None)#"U_{1000}^{(2)} - U_{100}^{(2)}")
    
    # Compute percent error across the map
    error = (grid_pred - grid_true)/ grid_true * 100.0
    print(np.average(error.total))

    plt.show()
if __name__ == "__main__":
    main()
