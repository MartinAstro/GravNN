import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os
from GravNN.Support.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Statistics import mean_std_median, sigma_mask

from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks.Data import training_validation_split
from GravNN.Support.StateObject import StateObject
from GravNN.Regression.Regression import Regression



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

def compute_stats(grid_true, grid_predict):
    #* Difference and stats
    diff = grid_pred - grid_true

    # This ensures the same features are being evaluated independent of what degree is taken off at beginning
    two_sigma_mask, two_sigma_mask_compliment = sigma_mask(grid_true.total, 2)
    three_sigma_mask, three_sigma_mask_compliment = sigma_mask(grid_true.total, 3)

    rse_stats = mean_std_median(diff.total, prefix='rse')
    sigma_2_stats = mean_std_median(diff.total, two_sigma_mask, "sigma_2")
    sigma_2_c_stats = mean_std_median(diff.total, two_sigma_mask_compliment, "sigma_2_c")
    sigma_3_stats = mean_std_median(diff.total, three_sigma_mask, "sigma_3")
    sigma_3_c_stats = mean_std_median(diff.total, three_sigma_mask_compliment, "sigma_3_c")

    extras = {
                'max_error' : [np.max(diff.total)]
                }

    entries = { **rse_stats,
                **sigma_2_stats,
                **sigma_2_c_stats,
                **sigma_3_stats,
                **sigma_3_c_stats,
                **extras
                    }
    return entries

def main():
    
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 100
    model_deg = 33

    # * Generate the true acceleration
    trajectory = RandomDist(planet, [planet.radius, planet.radius+420000.0], 1000000)
    x, a, u = get_sh_data(trajectory, model_file, max_deg=max_deg, deg_removed=2, override=False)
    x, a, u, x_val, a_val, u_val = training_validation_split(x, a, u, 9500, 500)
    grid_true = StateObject(trajectory=trajectory, accelerations=a)

    deg_list =  np.arange(3, model_deg, 1, dtype=int)#5, 101-3,dtype=int)
    df_all = pd.DataFrame()


    directory = os.path.join(os.path.abspath('.'), '/GravNN/Files/GravityModels/Regressed/')

    sh_df = pd.read_pickle("Data/Dataframes/regress_sh.data")
    nn_df = pd.read_pickle("Data/Dataframes/regress_nn.data")
    pinn_df = pd.read_pickle("Data/Dataframes/regress_pinn.data")

    df_file = "Data/Dataframes/sh_regress_stats.data"


    df_all = pd.DataFrame()
    for i in range(len(sh_df)):
        row = sh_df.iloc[i]
        file_name = row['model_identifier'] # TODO: Change this name
        deg = int(file_name.split("_")[1])

        #* Predict the value at the training data 
        x_est, a_est, u_est = get_sh_data(trajectory, directory + file_name +'.csv', max_deg=deg, deg_removed=2, override=False)
        x_est, a_est, u_est, x_val, a_val, u_val = training_validation_split(x_est, a_est, u_est, 9500, 500)
        grid_pred = StateObject(trajectory=trajectory, accelerations=a_est)

        entries = compute_stats(grid_true, grid_pred)
        entries.update({'deg' : [deg],
                        'params' : [deg*(deg+1)]})
        

        df = pd.DataFrame().from_dict(entries)#.set_index(sh_df.index[i])
        df_all = df_all.append(df)
    
    df_all.index = sh_df.index
    sh_df = sh_df.join(df_all)
    sh_df.to_pickle(df_file)


if __name__ == "__main__":
    main()

