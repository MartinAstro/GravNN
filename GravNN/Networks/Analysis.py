
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Networks import utils
from GravNN.Support.Grid import Grid
from GravNN.Support.StateObject import StateObject
from GravNN.Support.transformations import sphere2cart, cart2sph, invert_projection, project_acceleration
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Support.Statistics import mean_std_median, sigma_mask

import numpy as np
import pickle
import time
import os
import pandas as pd

class Analysis():
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.x_transformer = config['x_transformer'][0]
        self.a_transformer = config['a_transformer'][0]

    def compute_rse_stats(self, test_trajectories):
        stats = {}

        for name, map_traj in test_trajectories.items():
            model_file = map_traj.celestial_body.sh_hf_file
            x, a, u = get_sh_data(map_traj, model_file, self.config['max_deg'][0], self.config['deg_removed'][0])

            if self.config['basis'][0] == 'spherical':
                x = cart2sph(x)
                a = project_acceleration(x, a)
                x[:,1:3] = np.deg2rad(x[:,1:3])

            x = self.x_transformer.transform(x)
            a = self.a_transformer.transform(a)

            U_pred, acc_pred = self.model.predict(x.astype('float32'))

            x = self.x_transformer.inverse_transform(x)
            a = self.a_transformer.inverse_transform(a)
            acc_pred = self.a_transformer.inverse_transform(acc_pred)

            if self.config['basis'][0] == 'spherical':
                x[:,1:3] = np.rad2deg(x[:,1:3])
                #x = sphere2cart(x)
                a = invert_projection(x, a)
                a_pred = invert_projection(x, acc_pred.astype(float))# numba requires that the types are the same 

            grid_true = Grid(trajectory=map_traj, accelerations=a)
            grid_pred = Grid(trajectory=map_traj, accelerations=acc_pred)
            diff = grid_pred - grid_true
        
            # This ensures the same features are being evaluated independent of what degree is taken off at beginning
            two_sigma_mask, two_sigma_mask_compliment = sigma_mask(grid_true.total, 2)
            three_sigma_mask, three_sigma_mask_compliment = sigma_mask(grid_true.total, 3)

            rse_stats = mean_std_median(diff.total, prefix=name+'_rse')
            sigma_2_stats = mean_std_median(diff.total, two_sigma_mask, name+"_sigma_2")
            sigma_2_c_stats = mean_std_median(diff.total, two_sigma_mask_compliment, name+"_sigma_2_c")
            sigma_3_stats = mean_std_median(diff.total, three_sigma_mask, name+"_sigma_3")
            sigma_3_c_stats = mean_std_median(diff.total, three_sigma_mask_compliment, name+"_sigma_3_c")
            
            extras = {
                name+'_max_error' : [np.max(diff.total)]
            }

            map_stats = { **rse_stats,
                    **sigma_2_stats,
                    **sigma_2_c_stats,
                    **sigma_3_stats,
                    **sigma_3_c_stats,
                    **extras
                    }

            # Compute nearest SH degree
            with open(self.config['sh_truth'][0]+name+".data", 'rb') as f:
                stats_df = pickle.load(f)

            sh_stats = {
                name+'_sh_rse_mean' : [self.nearest_sh(stats_df['rse_mean'], map_stats[name+'_rse_mean'])],
                name+'_sh_rse_median' : [self.nearest_sh(stats_df['rse_median'], map_stats[name+'_rse_median'])],

                name+'_sh_sigma_2_mean' : [self.nearest_sh(stats_df['sigma_2_mean'], map_stats[name+'_sigma_2_mean'])],
                name+'_sh_sigma_2_median' : [self.nearest_sh(stats_df['sigma_2_median'], map_stats[name+'_sigma_2_median'])],
                name+'_sh_sigma_2_c_mean' : [self.nearest_sh(stats_df['sigma_2_c_mean'], map_stats[name+'_sigma_2_c_mean'])],
                name+'_sh_sigma_2_c_median' : [self.nearest_sh(stats_df['sigma_2_c_median'], map_stats[name+'_sigma_2_c_median'])],

                name+'_sh_sigma_3_mean' : [self.nearest_sh(stats_df['sigma_3_mean'], map_stats[name+'_sigma_3_mean'])],
                name+'_sh_sigma_3_median' : [self.nearest_sh(stats_df['sigma_3_median'], map_stats[name+'_sigma_3_median'])],
                name+'_sh_sigma_3_c_mean' : [self.nearest_sh(stats_df['sigma_3_c_mean'], map_stats[name+'_sigma_3_c_mean'])],
                name+'_sh_sigma_3_c_median' : [self.nearest_sh(stats_df['sigma_3_c_median'], map_stats[name+'_sigma_3_c_median'])]
            }
            stats.update(map_stats)
            stats.update(sh_stats)
        return stats

    def compute_alt_stats(self, planet, density_deg):
        alt_list = np.linspace(0, 500000, 100, dtype=float) # Every 0.5 kilometers above surface
        window = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 35, 45, 100, 200, 300, 400]) # Close to surface distribution
        LEO_window_upper = window + 420000 # Window around LEO
        LEO_window_lower = -1.0*window + 420000
        alt_list = np.concatenate([alt_list, window, LEO_window_lower, LEO_window_upper])
        alt_list = np.sort(np.unique(alt_list))
        #alt_list =  np.array([0, 1, 2, 3, 4, 5])
        stats = {}
        df_all = pd.DataFrame()
        for alt in alt_list: 
            trajectory = DHGridDist(planet, planet.radius + alt, degree=density_deg)
            model_file = trajectory.celestial_body.sh_hf_file
            x, a, u = get_sh_data(trajectory, model_file, self.config['max_deg'][0], 2)

            if self.config['basis'][0] == 'spherical':
                x = cart2sph(x)
                a = project_acceleration(x, a)
                x[:,1:3] = np.deg2rad(x[:,1:3])

            x = self.x_transformer.transform(x)
            a = self.a_transformer.transform(a)

            U_pred, acc_pred = self.model.predict(x.astype('float32'))

            x = self.x_transformer.inverse_transform(x)
            a = self.a_transformer.inverse_transform(a)
            acc_pred = self.a_transformer.inverse_transform(acc_pred)

            if self.config['basis'][0] == 'spherical':
                x[:,1:3] = np.rad2deg(x[:,1:3])
                #x = sphere2cart(x)
                a = invert_projection(x, a)
                a_pred = invert_projection(x, acc_pred.astype(float))# numba requires that the types are the same 

            grid_true = Grid(trajectory=trajectory, accelerations=a)
            grid_pred = Grid(trajectory=trajectory, accelerations=acc_pred)
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
                    'alt' : [alt], 
                    'max_error' : [np.max(diff.total)]
                }
            entries = { **rse_stats,
                    **sigma_2_stats,
                    **sigma_2_c_stats,
                    **sigma_3_stats,
                    **sigma_3_c_stats,
                    **extras
                    }
            stats.update(entries)
            df = pd.DataFrame().from_dict(stats).set_index('alt')
            df_all = df_all.append(df)
        print(df_all)
        return df_all


    def compute_bennu_rse_stats(self, test_trajectories):
        stats = {}

        for name, map_traj in test_trajectories.items():
            model_file = map_traj.celestial_body.obj_hf_file
            x, a, u = get_poly_data(map_traj, model_file)

            if self.config['basis'][0] == 'spherical':
                x = cart2sph(x)
                a = project_acceleration(x, a)
                x[:,1:3] = np.deg2rad(x[:,1:3])

            x = self.x_transformer.transform(x)
            a = self.a_transformer.transform(a)

            U_pred, acc_pred = self.model.predict(x.astype('float32'))

            x = self.x_transformer.inverse_transform(x)
            a = self.a_transformer.inverse_transform(a)
            acc_pred = self.a_transformer.inverse_transform(acc_pred)

            if self.config['basis'][0] == 'spherical':
                x[:,1:3] = np.rad2deg(x[:,1:3])
                #x = sphere2cart(x)
                a = invert_projection(x, a)
                a_pred = invert_projection(x, acc_pred.astype(float))# numba requires that the types are the same 

            grid_true = StateObject(trajectory=map_traj, accelerations=a)
            grid_pred = StateObject(trajectory=map_traj, accelerations=acc_pred)
            diff = grid_pred - grid_true
        
            # This ensures the same features are being evaluated independent of what degree is taken off at beginning
            two_sigma_mask, two_sigma_mask_compliment = sigma_mask(grid_true.total, 2)
            three_sigma_mask, three_sigma_mask_compliment = sigma_mask(grid_true.total, 3)

            rse_stats = mean_std_median(diff.total, prefix=name+'_rse')
            sigma_2_stats = mean_std_median(diff.total, two_sigma_mask, name+"_sigma_2")
            sigma_2_c_stats = mean_std_median(diff.total, two_sigma_mask_compliment, name+"_sigma_2_c")
            sigma_3_stats = mean_std_median(diff.total, three_sigma_mask, name+"_sigma_3")
            sigma_3_c_stats = mean_std_median(diff.total, three_sigma_mask_compliment, name+"_sigma_3_c")
            
            extras = {
                name+'_max_error' : [np.max(diff.total)]
            }

            map_stats = { **rse_stats,
                    **sigma_2_stats,
                    **sigma_2_c_stats,
                    **sigma_3_stats,
                    **sigma_3_c_stats,
                    **extras
                    }

            # # Compute nearest SH degree
            # with open(self.config['sh_truth'][0]+name+".data", 'rb') as f:
            #     stats_df = pickle.load(f)

            # sh_stats = {
            #     name+'_sh_rse_mean' : [self.nearest_sh(stats_df['rse_mean'], map_stats[name+'_rse_mean'])],
            #     name+'_sh_rse_median' : [self.nearest_sh(stats_df['rse_median'], map_stats[name+'_rse_median'])],

            #     name+'_sh_sigma_2_mean' : [self.nearest_sh(stats_df['sigma_2_mean'], map_stats[name+'_sigma_2_mean'])],
            #     name+'_sh_sigma_2_median' : [self.nearest_sh(stats_df['sigma_2_median'], map_stats[name+'_sigma_2_median'])],
            #     name+'_sh_sigma_2_c_mean' : [self.nearest_sh(stats_df['sigma_2_c_mean'], map_stats[name+'_sigma_2_c_mean'])],
            #     name+'_sh_sigma_2_c_median' : [self.nearest_sh(stats_df['sigma_2_c_median'], map_stats[name+'_sigma_2_c_median'])],

            #     name+'_sh_sigma_3_mean' : [self.nearest_sh(stats_df['sigma_3_mean'], map_stats[name+'_sigma_3_mean'])],
            #     name+'_sh_sigma_3_median' : [self.nearest_sh(stats_df['sigma_3_median'], map_stats[name+'_sigma_3_median'])],
            #     name+'_sh_sigma_3_c_mean' : [self.nearest_sh(stats_df['sigma_3_c_mean'], map_stats[name+'_sigma_3_c_mean'])],
            #     name+'_sh_sigma_3_c_median' : [self.nearest_sh(stats_df['sigma_3_c_median'], map_stats[name+'_sigma_3_c_median'])]
            # }
            stats.update(map_stats)
            #stats.update(sh_stats)
        return stats

    def compute_bennu_alt_stats(self, planet, density_deg):
        alt_list = np.linspace(0, 1500, 10, dtype=float) # Every 0.5 kilometers above surface
        alt_list = np.sort(np.unique(alt_list))
        stats = {}
        df_all = pd.DataFrame()
        for alt in alt_list: 
            trajectory = DHGridDist(planet, planet.radius + alt, degree=density_deg)
            model_file = trajectory.celestial_body.obj_hf_file
            x, a, u = get_poly_data(trajectory, model_file)

            if self.config['basis'][0] == 'spherical':
                x = cart2sph(x)
                a = project_acceleration(x, a)
                x[:,1:3] = np.deg2rad(x[:,1:3])

            x = self.x_transformer.transform(x)
            a = self.a_transformer.transform(a)

            U_pred, acc_pred = self.model.predict(x.astype('float32'))

            x = self.x_transformer.inverse_transform(x)
            a = self.a_transformer.inverse_transform(a)
            acc_pred = self.a_transformer.inverse_transform(acc_pred)

            if self.config['basis'][0] == 'spherical':
                x[:,1:3] = np.rad2deg(x[:,1:3])
                #x = sphere2cart(x)
                a = invert_projection(x, a)
                a_pred = invert_projection(x, acc_pred.astype(float))# numba requires that the types are the same 

            grid_true = Grid(trajectory=trajectory, accelerations=a)
            grid_pred = Grid(trajectory=trajectory, accelerations=acc_pred)
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
                    'alt' : [alt], 
                    'max_error' : [np.max(diff.total)]
                }
            entries = { **rse_stats,
                    **sigma_2_stats,
                    **sigma_2_c_stats,
                    **sigma_3_stats,
                    **sigma_3_c_stats,
                    **extras
                    }
            stats.update(entries)
            df = pd.DataFrame().from_dict(stats).set_index('alt')
            df_all = df_all.append(df)
        print(df_all)
        return df_all

    def nearest_sh(self, map_stat_series, value):
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
        
        nearest_sh = np.round(line_x[i])
        return nearest_sh

        