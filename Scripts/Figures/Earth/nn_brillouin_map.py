        
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories import DHGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Networks.Model import load_config_and_model

def plot(df, idx, planet, trajectory):

    map_vis = MapVisualization('m/s^2')
    map_vis.fig_size = map_vis.full_page_default
    map_vis.tick_interval = [60, 60]

    grav_model = SphericalHarmonics(planet.sh_file,1000,trajectory).load()
    a_1000 = grav_model.accelerations
    grid_true = Grid(trajectory=trajectory, accelerations=a_1000)

    grav_model = SphericalHarmonics(planet.sh_file,55,trajectory).load()
    a_sh = grav_model.accelerations
    grid_sh = Grid(trajectory=trajectory, accelerations=a_sh)

    grav_model = SphericalHarmonics(planet.sh_file,2,trajectory).load()
    a_2 = grav_model.accelerations
    grid_a2 = Grid(trajectory=trajectory, accelerations=a_2)

    row = df.iloc[idx]
    model_id = row['id']
    config, model = load_config_and_model(model_id, df)

    # The PINN Model
    a_pred = model._compute_acceleration(trajectory.positions)
    grid_pred = Grid(trajectory=trajectory, accelerations=a_pred) 

    # If the PINN model includes point mass + J2, remove it
    if config['deg_removed'][0] == -1:
        grid_pred -= grid_a2
    grid_true -= grid_a2
    grid_sh -= grid_a2
        

    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    vlim = [grid_pred.total.min(), np.mean(grid_pred.total) + 2*np.std(grid_pred.total)]
    map_vis.plot_grid(grid_pred.total, vlim=vlim, label=None, new_fig=False)
    plt.gcf().get_axes()[-2].set_title("PINN Model no Vlim")
    
    plt.subplot(2,1,2)
    map_vis.plot_grid(grid_true.total, vlim=vlim, label=None, new_fig=False)
    plt.gcf().get_axes()[-2].set_title("True Model")
    

    plt.figure(figsize=(8,6))
    plt.subplot(2,2,1)
    vlim = [grid_pred.total.min(), grid_pred.total.max()]
    map_vis.plot_grid(grid_pred.total, vlim=vlim, label=None, new_fig=False)
    plt.gcf().get_axes()[-2].set_title("PINN Model a2")
    
    plt.subplot(2,2,3)
    map_vis.plot_grid(grid_sh.total, vlim=vlim, label=None, new_fig=False)
    plt.gcf().get_axes()[-2].set_title("SH Model a2")

    plt.subplot(2,2,2)
    PINN_grid_difference = grid_pred - grid_true
    map_vis.plot_grid(PINN_grid_difference.total, vlim=vlim, label=None, new_fig=False)
    plt.gcf().get_axes()[-2].set_title("PINN Model Diff")
    
    plt.subplot(2,2,4)
    SH_grid_difference = grid_sh - grid_true
    map_vis.plot_grid(SH_grid_difference.total, vlim=vlim, label=None, new_fig=False)
    plt.gcf().get_axes()[-2].set_title("SH Model Diff")



    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    error = PINN_grid_difference / grid_true * 100.0
    map_vis.plot_grid(error.total, vlim=[0,100], label=None, new_fig=False)
    print(f"PINN Percent Error: {np.average(error.total)}")
    print(f"PINN RMS Error: {np.average(PINN_grid_difference.total)}")
    plt.gcf().get_axes()[-2].set_title("Error PINN")

    plt.subplot(2,1,2)
    error = SH_grid_difference / grid_true * 100.0
    map_vis.plot_grid(error.total, vlim=[0,100], label=None, new_fig=False)
    print(f"SH Percent Error: {np.average(error.total)}")
    print(f"SH RMS Error: {np.average(SH_grid_difference.total)}")
    plt.gcf().get_axes()[-2].set_title("Error SH")


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

    surface_data = DHGridDist(planet, planet.radius, degree=density_deg)
    plot(df, idx, planet, surface_data)
    
    # LEO_data = DHGridDist(planet, planet.radius+420000, degree=density_deg)
    # plot(df, idx, planet, LEO_data)
    
    # high_alt_data = DHGridDist(planet, planet.radius*10, degree=density_deg)
    # plot(df, idx, planet, high_alt_data)


    plt.show()
if __name__ == "__main__":
    main()
