
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Networks import utils
from GravNN.Support.Grid import Grid
from GravNN.Support.StateObject import StateObject
from GravNN.Support.transformations import sphere2cart, cart2sph, invert_projection, project_acceleration
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.FibonacciDist import FibonacciDist
from GravNN.Support.Statistics import mean_std_median, sigma_mask

import numpy as np
import pickle
import time
import os
import pandas as pd

def nearest_analytic(map_stat_series, value):
    i = 0
    if value < map_stat_series.iloc[i]: 
        while value < map_stat_series.iloc[i]:
            i += 1
            if i >= len(map_stat_series)-1:
                return -1
    else:
        return -1
    upper_y = map_stat_series.iloc[i-1]
    lower_y = map_stat_series.iloc[i]

    upper_x = map_stat_series.index[i-1] #x associated with upper bound
    lower_x = map_stat_series.index[i] # x associated with lower bound

    slope = (lower_y - upper_y)/(lower_x - upper_x)
    
    line_x = np.linspace(upper_x, lower_x, 100)
    line_y = slope*(line_x - upper_x) + upper_y

    i=0
    while value < line_y[i]:
        i += 1
    
    nearest_param = np.round(line_x[i])
    return nearest_param

def diff_map_and_stats(name, trajectory, a, acc_pred):
    state_obj_true = StateObject(trajectory=trajectory, accelerations=a)
    state_obj_pred = StateObject(trajectory=trajectory, accelerations=acc_pred)
    diff = state_obj_pred - state_obj_true

    # This ensures the same features are being evaluated independent of what degree is taken off at beginning
    two_sigma_mask, two_sigma_mask_compliment = sigma_mask(state_obj_true.total, 2)
    three_sigma_mask, three_sigma_mask_compliment = sigma_mask(state_obj_true.total, 3)

    rse_stats = mean_std_median(diff.total, prefix=name+'_rse')
    sigma_2_stats = mean_std_median(diff.total, two_sigma_mask, name+"_sigma_2")
    sigma_2_c_stats = mean_std_median(diff.total, two_sigma_mask_compliment, name+"_sigma_2_c")
    sigma_3_stats = mean_std_median(diff.total, three_sigma_mask, name+"_sigma_3")
    sigma_3_c_stats = mean_std_median(diff.total, three_sigma_mask_compliment, name+"_sigma_3_c")

    stats = {
        **rse_stats,
        **sigma_2_stats,
        **sigma_2_c_stats,
        **sigma_3_stats,
        **sigma_3_c_stats
    }
    return diff, stats

class Analysis():
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.x_transformer = config['x_transformer'][0]
        self.a_transformer = config['a_transformer'][0]

        if "Planet" in self.config['planet'][0].__module__:
            self.generate_analytic_data_fcn = get_sh_data
            self.body_type = "Planet"
        elif "Asteroid" in self.config['planet'][0].__module__:
            self.generate_analytic_data_fcn = get_poly_data
            self.body_type = "Asteroid"
        else:
            exit("The celestial body %s does not have a perscribed gravity model!" % (str(self.config['planet'][0]),))

    
    def generate_nn_data(self, x, a):
        if self.config['basis'][0] == 'spherical':
            x = cart2sph(x)
            a = project_acceleration(x, a)
            x[:,1:3] = np.deg2rad(x[:,1:3])

        x = self.x_transformer.transform(x)
        a = self.a_transformer.transform(a)

        U_pred, acc_pred, laplace, curl = self.model.predict(x.astype('float32'))

        x = self.x_transformer.inverse_transform(x)
        a = self.a_transformer.inverse_transform(a)
        acc_pred = self.a_transformer.inverse_transform(acc_pred)

        if self.config['basis'][0] == 'spherical':
            x[:,1:3] = np.rad2deg(x[:,1:3])
            #x = sphere2cart(x)
            a = invert_projection(x, a)
            acc_pred = invert_projection(x, acc_pred.astype(float))# numba requires that the types are the same 
        return acc_pred
    


    def compute_nearest_analytic(self, name, map_stats):
        # Compute nearest SH degree
        with open("Data/Dataframes/" + self.config['analytic_truth'][0]+name+".data", 'rb') as f:
            stats_df = pickle.load(f)

        param_stats = {
            name+'_param_rse_mean' : [nearest_analytic(stats_df['rse_mean'], map_stats[name+'_rse_mean'])],
            name+'_param_rse_median' : [nearest_analytic(stats_df['rse_median'], map_stats[name+'_rse_median'])],

            name+'_param_sigma_2_mean' : [nearest_analytic(stats_df['sigma_2_mean'], map_stats[name+'_sigma_2_mean'])],
            name+'_param_sigma_2_median' : [nearest_analytic(stats_df['sigma_2_median'], map_stats[name+'_sigma_2_median'])],
            name+'_param_sigma_2_c_mean' : [nearest_analytic(stats_df['sigma_2_c_mean'], map_stats[name+'_sigma_2_c_mean'])],
            name+'_param_sigma_2_c_median' : [nearest_analytic(stats_df['sigma_2_c_median'], map_stats[name+'_sigma_2_c_median'])],

            name+'_param_sigma_3_mean' : [nearest_analytic(stats_df['sigma_3_mean'], map_stats[name+'_sigma_3_mean'])],
            name+'_param_sigma_3_median' : [nearest_analytic(stats_df['sigma_3_median'], map_stats[name+'_sigma_3_median'])],
            name+'_param_sigma_3_c_mean' : [nearest_analytic(stats_df['sigma_3_c_mean'], map_stats[name+'_sigma_3_c_mean'])],
            name+'_param_sigma_3_c_median' : [nearest_analytic(stats_df['sigma_3_c_median'], map_stats[name+'_sigma_3_c_median'])]
        }
        return param_stats

    def compute_rse_stats(self, test_trajectories):
        stats = {}

        for name, map_traj in test_trajectories.items():
            x, a, u = self.generate_analytic_data_fcn(map_traj, self.config['grav_file'][0] , **self.config)

            acc_pred = self.generate_nn_data(x, a)
            diff, diff_stats = diff_map_and_stats(name, map_traj, a, acc_pred)
            
            map_stats = { 
                    **diff_stats,
                    name+'_max_error' : [np.max(diff.total)]
                    }

            analytic_neighbors = self.compute_nearest_analytic(name, map_stats)
            stats.update(map_stats)
            stats.update(analytic_neighbors)
        return stats

    def compute_alt_stats(self, planet, altitudes, points):
        stats = {}
        df_all = pd.DataFrame()
        for alt in altitudes: 
            trajectory = FibonacciDist(planet, planet.radius + alt, points)
            model_file = trajectory.celestial_body.sh_hf_file
            x, a, u = get_sh_data(trajectory, model_file, **self.config)
            
            acc_pred = self.generate_nn_data(x, a)
            diff, diff_stats = diff_map_and_stats("", trajectory, a, acc_pred)

            extras = {
                    'alt' : [alt], 
                    'max_error' : [np.max(diff.total)]
                }
            entries = { 
                    **diff_stats,
                    **extras
                    }
            stats.update(entries)
            df = pd.DataFrame().from_dict(stats).set_index('alt')
            df_all = df_all.append(df)
        print(df_all)
        return df_all
        