import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

from GravNN.Support.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Support.Statistics import mean_std_median, sigma_mask

from GravNN.Trajectories import DHGridDist, RandomDist
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks.Data import training_validation_split
from GravNN.Support.StateObject import StateObject
from GravNN.Regression.Regression import Regression



np.random.seed(1234)
tf.random.set_seed(0)

def get_regress_data(model_file):
    # Generate the data
    planet = Earth()
    trajectory = RandomDist(planet, [planet.radius, planet.radius+420000.0], 1000000)
    x, a, u = get_sh_data(trajectory,planet.sh_hf_file, 1000, 2)
    x, a, u, x_val, a_val, u_val = training_validation_split(x, a, u, 9500, 500)
    return x, a

def regress_model(x, a, max_deg):
    planet = Earth()
    grav_file = 'C:\\Users\\John\\Documents\\Research\\ML_Gravity\\GravNN\\Files\\GravityModels\\Regressed\\regress_' + str(max_deg) + ".csv"
    regressor = Regression(max_deg, planet, x, a)
    coefficients = regressor.perform_regression(remove_deg=True)
    regressor.save(grav_file)



def main():
    
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000
    model_deg = 33

    # * Generate the true acceleration
    df_file = "Data/Dataframes/sh_regress_stats_"+str(model_deg)+"_Random.data"
    trajectory = RandomDist(planet, [planet.radius, planet.radius+420000.0], 1000000)
    x, a, u = get_sh_data(trajectory, model_file, max_deg=max_deg, deg_removed=2)
    x, a, u, x_val, a_val, u_val = training_validation_split(x, a, u, 9500, 500)
    grid_true = StateObject(trajectory=trajectory, accelerations=a)



    deg_list =  np.arange(3, model_deg, 1, dtype=int)#5, 101-3,dtype=int)
    df_all = pd.DataFrame()

    x_regress = x
    a_regress = a
    
    #regress_model(x_regress, a_regress, model_deg)
    for deg in deg_list:

        #* Generate Model
        regress_model(x_regress, a_regress, deg)

        #* Predict the value at the trainnig data 
        x_est, a_est, u_est = get_sh_data(trajectory, 'C:\\Users\\John\\Documents\\Research\\ML_Gravity\\GravNN\\Files\\GravityModels\\Regressed\\regress_' + str(deg) +'.csv', max_deg=deg, deg_removed=2)
        x_est, a_est, u_est, x_val, a_val, u_val = training_validation_split(x_est, a_est, u_est, 9500, 500)
        grid_pred = StateObject(trajectory=trajectory, accelerations=a_est)


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

        extras = {'deg' : [deg],
                  'max_error' : [np.max(diff.total)]
                  }

        entries = { **rse_stats,
                    **sigma_2_stats,
                    **sigma_2_c_stats,
                    **sigma_3_stats,
                    **sigma_3_c_stats,
                    **extras
                    }
        #for d in (rse_stats, percent_stats, percent_rel_stats, sigma_2_stats, sigma_2_c_stats, sigma_3_stats, sigma_3_c_stats): entries.update(d)
        
        df = pd.DataFrame().from_dict(entries).set_index('deg')
        df_all = df_all.append(df)
    
    df_all.to_pickle(df_file)


if __name__ == "__main__":
    main()

