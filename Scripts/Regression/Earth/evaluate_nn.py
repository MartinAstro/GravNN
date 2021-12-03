import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import (SphericalHarmonics,
                                                     get_sh_data)
from GravNN.Networks.Model import count_nonzero_params, load_config_and_model
from GravNN.Support.StateObject import StateObject
from GravNN.Support.Statistics import mean_std_median, sigma_mask
from GravNN.Trajectories import FibonacciDist, RandomDist

np.random.seed(1234)
tf.random.set_seed(0)
def get_nn_data(x, model, config):
    x_transformer = config['x_transformer'][0]
    a_transformer = config['a_transformer'][0]

    x = x_transformer.transform(x)
    a_pred = model.generate_acceleration(x.astype('float32'))

    x = x_transformer.inverse_transform(x)
    a_pred = a_transformer.inverse_transform(a_pred)

    return a_pred

def compute_stats(grid_true, grid_pred):
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

def compute_nn_regression_statistics(nn_df, nn_df_stats_file, model_df, trajectory, grid_true):
    x = trajectory.positions
    df_all = pd.DataFrame()
    model_df = pd.read_pickle(model_df)
    for i in range(len(nn_df)):
        row = nn_df.iloc[i]
        model_id = row['model_identifier'] # TODO: Change this name
        config, model = load_config_and_model(model_id, model_df)

        a_est = get_nn_data(x, model, config)
        grid_pred = StateObject(trajectory=trajectory, accelerations=a_est)

        entries = compute_stats(grid_true, grid_pred)
        entries.update({'params' : [count_nonzero_params(model)]})

        df = pd.DataFrame().from_dict(entries)#.set_index(nn_df.index[i])
        df_all = df_all.append(df)
    
    df_all.index = nn_df.index
    nn_df = nn_df.join(df_all)
    nn_df.to_pickle(nn_df_stats_file)

def main():
    """Given the regressed spherical harmonic and neural network models
    (generate_models_mp.py), compute the associated error of these
    regressed representations and store in new regress_stats dataframe. 
    """
    planet = Earth()
    model_file = planet.sh_hf_file
    max_deg = 1000

    # * Generate the true acceleration
    trajectory = RandomDist(planet, [planet.radius, planet.radius+420000.0], 10000)
    trajectory = FibonacciDist(planet, planet.radius, 250000)
    x, a, u = get_sh_data(trajectory, model_file, max_deg=max_deg, deg_removed=2, override=False)
    grid_true = StateObject(trajectory=trajectory, accelerations=a)

    # nn_df = pd.read_pickle("Data/Dataframes/regress_nn_3.data")
    # nn_df_stats_file = "Data/Dataframes/nn_regress_3_stats.data"
    # compute_nn_regression_statistics(nn_df, nn_df_stats_file, trajectory, grid_true)

    # pinn_df = pd.read_pickle("Data/Dataframes/regress_pinn_3.data")
    # pinn_df_stats_file = "Data/Dataframes/pinn_regress_3_stats.data"
    # compute_nn_regression_statistics(pinn_df, pinn_df_stats_file, trajectory, grid_true)

    nn_df = pd.read_pickle("Data/Dataframes/regress_nn_4.data")
    nn_df_stats_file = "Data/Dataframes/nn_regress_4_stats.data"
    model_df = "Data/Dataframes/regressed_models_4.data" #v3
    compute_nn_regression_statistics(nn_df, nn_df_stats_file, model_df, trajectory, grid_true)

    pinn_df = pd.read_pickle("Data/Dataframes/regress_pinn_4.data")
    pinn_df_stats_file = "Data/Dataframes/pinn_regress_4_stats.data"
    compute_nn_regression_statistics(pinn_df, pinn_df_stats_file, model_df, trajectory, grid_true)
if __name__ == "__main__":
    main()

