        
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

    map_vis = MapVisualization('m/s^2')
    map_vis.fig_size = map_vis.full_page
    map_vis.tick_interval = [60, 60]

    # vlim= [0, 30]
    vlim = None

    planet = Earth()
    density_deg = 180

    df_file ='C:\\Users\\John\\Documents\\Research\\ML_Gravity\\Data\\Dataframes\\pinn_df.data'
    df_file ='Data/Dataframes/earth_trajectory_v2.data'
    df_file ='Data/Dataframes/useless_072621_v2.data'
    df_file ='Data/Dataframes/test.data'

    df = pd.read_pickle(df_file)

    trajectory = DHGridDist(planet, planet.radius, degree=density_deg)
    # trajectory = DHGridDist(planet, planet.radius+420000, degree=density_deg)
    grav_model = SphericalHarmonics(planet.sh_file,1000,trajectory).load()
    a_1000 = grav_model.accelerations
    grid_true = Grid(trajectory=trajectory, accelerations=a_1000)

    grav_model = SphericalHarmonics(planet.sh_file,55,trajectory).load()
    a_sh = grav_model.accelerations
    grid_sh = Grid(trajectory=trajectory, accelerations=a_sh)

    grav_model = SphericalHarmonics(planet.sh_file,2,trajectory).load()
    a_2 = grav_model.accelerations
    grid_a2 = Grid(trajectory=trajectory, accelerations=a_2)


    i = -1
    row = df.iloc[i]
    model_id = row['id']
    config, model = load_config_and_model(model_id, df)

    # The PINN Model
    a_pred = model.compute_acceleration(trajectory.positions)
    grid_pred = Grid(trajectory=trajectory, accelerations=a_pred) 


    # If the PINN model includes point mass + J2, remove it
    if config['deg_removed'][0] == -1:
        grid_pred -= grid_a2
    grid_true -= grid_a2
    grid_sh -= grid_a2
        
    vlim = [grid_pred.total.min(), np.mean(grid_pred.total) + 2*np.std(grid_pred.total)]
    map_vis.plot_grid(grid_pred.total, vlim=vlim, label=None)
    plt.title("Pred Model a2 no Vlim")
    
    vlim = [grid_pred.total.min(), grid_pred.total.max()]
    map_vis.plot_grid(grid_pred.total, vlim=vlim, label=None)
    plt.title("Pred Model a2")
    
    map_vis.plot_grid(grid_true.total, vlim=vlim, label=None)
    plt.title("True Model a2")

    error = (grid_pred - grid_true)/ grid_true * 100.0
    map_vis.plot_grid(error.total, vlim=[0,100], label=None)
    print(np.average(error.total))
    plt.title("Error PINN")
    
    error = (grid_sh - grid_true)/ grid_true * 100.0
    map_vis.plot_grid(error.total, vlim=[0,100], label=None)
    print(np.average(error.total))
    plt.title("Error SH")


    plt.show()
if __name__ == "__main__":
    main()
