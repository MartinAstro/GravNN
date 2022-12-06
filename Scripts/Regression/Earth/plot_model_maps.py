import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import (SphericalHarmonics,
                                                     get_sh_data)
from GravNN.Networks.Model import count_nonzero_params, load_config_and_model
from GravNN.Support.Grid import Grid
from GravNN.Support.StateObject import StateObject
from GravNN.Support.Statistics import mean_std_median, sigma_mask
from GravNN.Trajectories import FibonacciDist, RandomDist, DHGridDist
from GravNN.Visualization.MapVisualization import MapVisualization
import matplotlib.pyplot as plt
np.random.seed(1234)
tf.random.set_seed(0)

def get_sh_data(trajectory, gravity_file, **kwargs):

    # Handle cases where the keyword wasn't properly wrapped as a list []
    try:
        max_deg = int(kwargs['max_deg'][0])
        deg_removed = int(kwargs['deg_removed'][0])
    except:
        max_deg = int(kwargs['max_deg'])
        deg_removed = int(kwargs['deg_removed'])

    Call_r0_gm = SphericalHarmonics(gravity_file, degree=max_deg, trajectory=trajectory)
    accelerations = Call_r0_gm.load(override=kwargs['override']).accelerations
    potentials = Call_r0_gm.potentials

    Clm_r0_gm = SphericalHarmonics(gravity_file, degree=deg_removed, trajectory=trajectory)
    accelerations_Clm = Clm_r0_gm.load(override=kwargs['override']).accelerations
    potentials_Clm = Clm_r0_gm.potentials
    
    x = Call_r0_gm.positions # position (N x 3)
    a = accelerations - accelerations_Clm
    u = np.array(potentials - potentials_Clm).reshape((-1,1)) # potential (N x 1)

    return x, a, u


def plot_sh_model_map(sh_df, trajectory, grid_true):
    map_vis = MapVisualization()
    directory = os.path.join(os.path.abspath('.') , 'GravNN','Files', 'GravityModels','Regressed', 'Earth')
    deg_list = []
    for i in range(len(sh_df)):
        row = sh_df.iloc[i]
        file_name = row['model_identifier'] 
        deg = int(file_name.split("_")[1])
        if deg in deg_list:
            continue
        else:
            deg_list.append(deg)
        #* Predict the value at the training data 
        x_est, a_est, u_est = get_sh_data(trajectory, directory + "\\" + file_name, max_deg=deg, deg_removed=2, override=True)
        grid_pred = Grid(trajectory=trajectory, accelerations=a_est)
        map_vis.plot_grid(grid_pred.total, label="Accelerations "+ str(deg))
    
    map_vis.plot_grid(grid_true.total, label="true", vlim=[0, np.max(grid_pred.total)])


def plot_nn_model_map(nn_df, trajectory, model_df, grid_true):
    map_vis = MapVisualization()
    directory = os.path.join(os.path.abspath('.') , 'GravNN','Files', 'GravityModels','Regressed', 'Earth')
    for i in range(len(nn_df)):
        row = nn_df.iloc[i]

        #* Predict the value at the training data 
        model_id = row['model_identifier'] # TODO: Change this name
        config, model = load_config_and_model(model_id, model_df)
        a_est = model.compute_acceleration(trajectory.positions)
        grid_pred = Grid(trajectory=trajectory, accelerations=a_est)
        map_vis.plot_grid(grid_pred.total, label="Accelerations" + str(config['num_units'][0]))
    
    map_vis.plot_grid(grid_true.total, label="true", vlim=[0, np.max(grid_pred.total)])


def main():
    planet = Earth()
    model_file = planet.sh_file
    max_deg = 1000

    # * Generate the true acceleration
    # trajectory = RandomDist(planet, [planet.radius, planet.radius+420000.0], 10000)
    
    trajectory = DHGridDist(planet, planet.radius, degree=180)
    x, a, u = get_sh_data(trajectory, model_file, max_deg=max_deg, deg_removed=2, override=False)
    grid_true = Grid(trajectory=trajectory, accelerations=a)

    sh_df = pd.read_pickle("Data/Dataframes/Regression/Earth_SH_regression_9500_v1.data")
    pinn_df = pd.read_pickle("Data/Dataframes/Regression/Earth_PINN_regression_9500_v1.data")
    model_df = pd.read_pickle("Data/Dataframes/Regression/Earth_ML_models_regression_9500_v1.data")

    plot_sh_model_map(sh_df, trajectory, grid_true)
    plot_nn_model_map(pinn_df, trajectory, model_df, grid_true)
    plt.show()

if __name__ == "__main__":
    main()