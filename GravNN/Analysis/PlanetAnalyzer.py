
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Networks import utils
from GravNN.Networks.Data import standardize_output
from GravNN.Support.StateObject import StateObject
from GravNN.Trajectories import FibonacciDist
from GravNN.Support.Statistics import mean_std_median, sigma_mask

import numpy as np
import pickle
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


def diff_map_and_stats(name, trajectory, a, acc_pred, stats=['mean', 'std', 'median']):
    state_obj_true = StateObject(trajectory=trajectory, accelerations=a)
    state_obj_pred = StateObject(trajectory=trajectory, accelerations=acc_pred)
    diff = state_obj_pred - state_obj_true

    # This ensures the same features are being evaluated independent of what degree is taken off at beginning
    one_sigma_mask, one_sigma_mask_compliment = sigma_mask(state_obj_true.total, 1)
    two_sigma_mask, two_sigma_mask_compliment = sigma_mask(state_obj_true.total, 2)
    three_sigma_mask, three_sigma_mask_compliment = sigma_mask(state_obj_true.total, 3)

    rse_stats = mean_std_median(diff.total, prefix=name+'_rse', stats_idx=stats)
    sigma_1_stats = mean_std_median(diff.total, one_sigma_mask, name+"_sigma_1", stats_idx=stats)
    sigma_1_c_stats = mean_std_median(diff.total, one_sigma_mask_compliment, name+"_sigma_1_c", stats_idx=stats)
    sigma_2_stats = mean_std_median(diff.total, two_sigma_mask, name+"_sigma_2", stats_idx=stats)
    sigma_2_c_stats = mean_std_median(diff.total, two_sigma_mask_compliment, name+"_sigma_2_c", stats_idx=stats)
    sigma_3_stats = mean_std_median(diff.total, three_sigma_mask, name+"_sigma_3", stats_idx=stats)
    sigma_3_c_stats = mean_std_median(diff.total, three_sigma_mask_compliment, name+"_sigma_3_c", stats_idx=stats)

    stats = {
        **rse_stats,
        **sigma_1_stats,
        **sigma_1_c_stats,       
        **sigma_2_stats,
        **sigma_2_c_stats,
        **sigma_3_stats,
        **sigma_3_c_stats
    }
    return diff, stats

class PlanetAnalyzer():
    def __init__(self, model, config):
        self.config = config
        self.model = model
        self.x_transformer = config['x_transformer'][0]
        self.u_transformer = config['u_transformer'][0]
        self.a_transformer = config['a_transformer'][0]
        self.body_type = "Planet"

    def compute_nearest_analytic(self, name, map_stats):
        # Compute nearest SH degree
        with open("Data/Dataframes/" + self.config['analytic_truth'][0]+name+".data", 'rb') as f:
            stats_df = pickle.load(f)

        param_stats = {
            name+'_param_rse_mean' : [nearest_analytic(stats_df['rse_mean'], map_stats[name+'_rse_mean'])],
            name+'_param_rse_median' : [nearest_analytic(stats_df['rse_median'], map_stats[name+'_rse_median'])],

            name+'_param_sigma_1_mean' : [nearest_analytic(stats_df['sigma_1_mean'], map_stats[name+'_sigma_1_mean'])],
            name+'_param_sigma_1_median' : [nearest_analytic(stats_df['sigma_1_median'], map_stats[name+'_sigma_1_median'])],
            name+'_param_sigma_1_c_mean' : [nearest_analytic(stats_df['sigma_1_c_mean'], map_stats[name+'_sigma_1_c_mean'])],
            name+'_param_sigma_1_c_median' : [nearest_analytic(stats_df['sigma_1_c_median'], map_stats[name+'_sigma_1_c_median'])],

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
            # SH Data and NN Data
            x, a, u = get_sh_data(map_traj, self.config['grav_file'][0] , **self.config)
            data_pred = self.model.generate_nn_data(x)
            acc_pred = data_pred['a']
            
            # Generate map statistics on sets A, F, and C (2 and 3 sigma)
            diff, diff_stats = diff_map_and_stats(name, map_traj, a, acc_pred)
            map_stats = { 
                    **diff_stats,
                    name+'_max_error' : [np.max(diff.total)]
                    }

            # Calculate the spherical harmonic degree that yields approximately the same statistics
            analytic_neighbors = self.compute_nearest_analytic(name, map_stats)
            stats.update(map_stats)
            stats.update(analytic_neighbors)
        return stats

    def compute_alt_stats(self, planet, altitudes, points, sh_alt_df):
        stats = {}
        df_all = pd.DataFrame()

        
        for alt in altitudes: 
            trajectory = FibonacciDist(planet, planet.radius + alt, points)
            model_file = trajectory.celestial_body.sh_hf_file
            x, a, u = get_sh_data(trajectory, model_file, **self.config)
            acc_pred = self.model.generate_nn_data(x)['a']

            diff, diff_stats = diff_map_and_stats("", trajectory, a, acc_pred, 'mean')
            extras = {
                    'alt' : [alt], 
                    'max_error' : [np.max(diff.total)]
                }
            entries = { 
                    **diff_stats,
                    **extras
                    }
            stats.update(entries)

            # Check for the nearest SH in altitude
            analytic_neighbors = {
                        'param_rse_mean' : [nearest_analytic(sh_alt_df.loc[alt]['rse_mean'], entries['_rse_mean'])],
                        'param_sigma_1_mean' : [nearest_analytic(sh_alt_df.loc[alt]['sigma_1_mean'], entries['_sigma_1_mean'],)],
                        'param_sigma_1_c_mean' : [nearest_analytic(sh_alt_df.loc[alt]['sigma_1_c_mean'], entries['_sigma_1_c_mean'],)],
                        'param_sigma_2_mean' : [nearest_analytic(sh_alt_df.loc[alt]['sigma_2_mean'], entries['_sigma_2_mean'],)],
                        'param_sigma_2_c_mean' : [nearest_analytic(sh_alt_df.loc[alt]['sigma_2_c_mean'], entries['_sigma_2_c_mean'])],
                        'param_sigma_3_mean' : [nearest_analytic(sh_alt_df.loc[alt]['sigma_3_mean'], entries['_sigma_3_mean'])],
                        'param_sigma_3_c_mean' : [nearest_analytic(sh_alt_df.loc[alt]['sigma_3_c_mean'], entries['_sigma_3_c_mean'])],
                    }
            stats.update(analytic_neighbors)
            df = pd.DataFrame().from_dict(stats).set_index('alt')
            df_all = df_all.append(df)
        print(df_all)
        df_all.to_pickle( "/Data/Networks/"+str(model_id)+"/rse_alt.data")
        return df_all
        