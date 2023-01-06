        
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Moon
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories import DHGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Networks.Model import load_config_and_model

def plot(df, idx, planet, trajectory):

    map_vis = MapVisualization('m/s^2')
    map_vis.fig_size = map_vis.full_page_golden
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
    a_pred = model.compute_acceleration(trajectory.positions)
    grid_pred = Grid(trajectory=trajectory, accelerations=a_pred) 

    # If the PINN model includes point mass + J2, remove it
    if config['deg_removed'][0] == -1:
        grid_pred -= grid_a2
    grid_true -= grid_a2
    grid_sh -= grid_a2
        
    vlim = [grid_pred.total.min(), grid_pred.total.max()]
    map_vis.plot_grid(grid_true.total, vlim=vlim, label=None, ticks=True, labels=False, loc='top', orientation='horizontal')
    # plt.gcf().get_axes()[-2].set_title("True Model")
    plt.xlabel("[$m/s^2$]")

    map_vis.fig_size = map_vis.half_page_golden
    map_vis.plot_grid(grid_pred.total, vlim=vlim, label=None, ticks=False, colorbar=False, labels=False)
    # plt.title("PINN MODEL")

    map_vis.plot_grid(grid_sh.total, vlim=vlim, label=None, ticks=False, colorbar=False, labels=False)
    # plt.title("SH MODEL")


    PINN_grid_difference = grid_pred - grid_true
    SH_grid_difference = grid_sh - grid_true

    error = SH_grid_difference / grid_true * 100.0
    print(f"SH Percent Error: {np.average(error.total)}")
    print(f"SH RMS Error: {np.average(SH_grid_difference.total)}")

    PINN_grid_difference = grid_pred - grid_true
    error = PINN_grid_difference / grid_true * 100.0
    print(f"PINN Percent Error: {np.average(error.total)}")
    print(f"PINN RMS Error: {np.average(PINN_grid_difference.total)}")



def main():
    # Plotting 
    directory = os.path.abspath('.') +"/Plots/"
    os.makedirs(directory, exist_ok=True)

    planet = Moon()
    density_deg = 180

    df_file, idx = "Data/Dataframes/moon_I_III_SIRENS_test.data", -1
    df = pd.read_pickle(df_file)

    # SIRENS III
    surface_data = DHGridDist(planet, planet.radius, degree=density_deg)
    plot(df, idx, planet, surface_data)

    # III
    surface_data = DHGridDist(planet, planet.radius, degree=density_deg)
    plot(df, -3, planet, surface_data)

    plt.figure(1)
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Moon_Truth.pdf", pad_inches=0.0)
    plt.figure(2)
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Moon_SIRENS.pdf", pad_inches=0.0)
    plt.figure(3)
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Moon_SH.pdf", pad_inches=0.0)
    plt.figure(5)
    plt.tight_layout()
    plt.savefig("Plots/PINNIII/Moon_PINN.pdf", pad_inches=0.0)

    


    plt.show()
if __name__ == "__main__":
    main()
