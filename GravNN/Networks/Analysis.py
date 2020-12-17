
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Networks import utils
from GravNN.Support.Grid import Grid
from GravNN.Support.transformations import sphere2cart, cart2sph, invert_projection, project_acceleration

import numpy as np
import pickle
import time
import os
import pandas as pd

class Analysis():
    def __init__(self, model_file, map_trajectories, x_transformer, a_transformer, config):
        self.model_file = model_file
        self.map_trajectories = map_trajectories
        self.config = config
        self.x_transformer = x_transformer
        self.a_transformer = a_transformer
        self.stats = {}
        self.stats.update(config)

    def __call__(self, model, history, df_file, save=False):
        timestamp = pd.Timestamp(time.time(), unit='s').round('s').ctime()

        self.testing_stats(model)
        self.model_size_stats(model)

        if history is None:
            print("History object was none, nothing to save here")
            return 
        
        self.stats.update({'timetag': [timestamp],
                           'history' : [history.history],
                           'train_time' : [history.history['time_delta']],
                           'directory' : [None]
                           })

        if save:
            utils.save_dataframe_row(self.stats, df_file)
            
            self.directory = os.path.abspath('.') +"/Plots/"+ str(pd.Timestamp(timestamp).to_julian_date()) + "/"
            os.makedirs(self.directory, exist_ok=True)
            model.network.save(self.directory + "network")
            self.stats.update({'directory' : [self.directory]})

    def model_size_stats(self, model):
        size_stats = {
            'params' : [utils.count_nonzero_params(model)],
            'size' : [utils.get_gzipped_model_size(model)],
        }
        self.stats.update(size_stats)
        return size_stats

    def testing_stats(self, model):
        for name, map_traj in self.map_trajectories.items():
            Call_r0_gm = SphericalHarmonics(self.model_file, degree=int(self.config['max_deg'][0]), trajectory=map_traj)
            Call_a = Call_r0_gm.load()
            
            Clm_r0_gm = SphericalHarmonics(self.model_file, degree=int(self.config['deg_removed'][0]), trajectory=map_traj)
            Clm_a = Clm_r0_gm.load()

            x = Call_r0_gm.positions # position (N x 3)
            a = Call_a - Clm_a

            if self.config['basis'][0] == 'spherical':
                x = cart2sph(x)
                a = project_acceleration(x, a)
                x[:,1:3] = np.deg2rad(x[:,1:3])

            x = self.x_transformer.transform(x)
            a = self.a_transformer.transform(a)

            U_pred, acc_pred = model.predict(x.astype('float32'))

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
            C22_r0_gm = SphericalHarmonics(self.model_file, degree=2, trajectory=map_traj)
            C22_a = C22_r0_gm.load()
            grid_C22 = Grid(trajectory=map_traj, accelerations=Call_a - C22_a)

            two_sigma_mask = np.where(grid_C22.total > (np.mean(grid_C22.total) + 2*np.std(grid_C22.total)))
            two_sigma_mask_compliment = np.where(grid_C22.total < (np.mean(grid_C22.total) + 2*np.std(grid_C22.total)))
            two_sig_features = diff.total[two_sigma_mask]
            two_sig_features_comp = diff.total[two_sigma_mask_compliment]

            three_sigma_mask = np.where(grid_C22.total > (np.mean(grid_C22.total) + 3*np.std(grid_C22.total)))
            three_sigma_mask_compliment = np.where(grid_C22.total < (np.mean(grid_C22.total) + 3*np.std(grid_C22.total)))
            three_sig_features = diff.total[three_sigma_mask]
            three_sig_features_comp = diff.total[three_sigma_mask_compliment]

            map_stats = {
                name+'_diff_mean' : [np.mean(np.sqrt(np.square(diff.total)))],
                name+'_diff_std' : [np.std(np.sqrt(np.square(diff.total)))],
                name+'_diff_median' : [np.median(np.sqrt(np.square(diff.total)))],

                name+'_sigma_2_mean' : [np.mean(np.sqrt(np.square(two_sig_features)))],
                name+'_sigma_2_std' : [np.std(np.sqrt(np.square(two_sig_features)))],
                name+'_sigma_2_median' : [np.median(np.sqrt(np.square(two_sig_features)))],

                name+'_sigma_2_c_mean' : [np.mean(np.sqrt(np.square(two_sig_features_comp)))],
                name+'_sigma_2_c_std' : [np.std(np.sqrt(np.square(two_sig_features_comp)))],
                name+'_sigma_2_c_median' : [np.median(np.sqrt(np.square(two_sig_features_comp)))],

                name+'_sigma_3_mean' : [np.mean(np.sqrt(np.square(three_sig_features)))],
                name+'_sigma_3_std' : [np.std(np.sqrt(np.square(three_sig_features)))],
                name+'_sigma_3_median' : [np.median(np.sqrt(np.square(three_sig_features)))],

                name+'_sigma_3_c_mean' : [np.mean(np.sqrt(np.square(three_sig_features_comp)))],
                name+'_sigma_3_c_std' : [np.std(np.sqrt(np.square(three_sig_features_comp)))],
                name+'_sigma_3_c_median' : [np.median(np.sqrt(np.square(three_sig_features_comp)))],

                name+'_max_error' : [np.max(np.sqrt(np.square(diff.total)))]
            }
            self.stats.update(map_stats)

            # Compute nearest SH degree
            with open(self.config['sh_truth'][0]+name+".data", 'rb') as f:
                stats_df = pickle.load(f)

            sh_stats = {
                name+'_sh_diff_mean' : [self.nearest_sh(stats_df['rse_mean'], map_stats[name+'_diff_mean'])],
                name+'_sh_diff_median' : [self.nearest_sh(stats_df['rse_median'], map_stats[name+'_diff_median'])],

                name+'_sh_sigma_2_mean' : [self.nearest_sh(stats_df['sigma_2_mean'], map_stats[name+'_sigma_2_mean'])],
                name+'_sh_sigma_2_median' : [self.nearest_sh(stats_df['sigma_2_median'], map_stats[name+'_sigma_2_median'])],
                name+'_sh_sigma_2_c_mean' : [self.nearest_sh(stats_df['sigma_2_c_mean'], map_stats[name+'_sigma_2_c_mean'])],
                name+'_sh_sigma_2_c_median' : [self.nearest_sh(stats_df['sigma_2_c_median'], map_stats[name+'_sigma_2_c_median'])],

                name+'_sh_sigma_3_mean' : [self.nearest_sh(stats_df['sigma_3_mean'], map_stats[name+'_sigma_3_mean'])],
                name+'_sh_sigma_3_median' : [self.nearest_sh(stats_df['sigma_3_median'], map_stats[name+'_sigma_3_median'])],
                name+'_sh_sigma_3_c_mean' : [self.nearest_sh(stats_df['sigma_3_c_mean'], map_stats[name+'_sigma_3_c_mean'])],
                name+'_sh_sigma_3_c_median' : [self.nearest_sh(stats_df['sigma_3_c_median'], map_stats[name+'_sigma_3_c_median'])]
            }
        
            self.stats.update(sh_stats)
        return map_stats

    def altitude_stats(self, model):
        deg_list =  [2, 25, 50, 75, 100, 150, 200]
        alt_list = np.linspace(planet.radius, planet.radius+500000, 100, dtype=float) # Every 0.5 kilometers
        alt_list = alt_list.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 35, 45, 100, 200, 300, 400]) # In the training range and the outskirts 
        alt_list = alt_list.append([420000-400, 420000-300, 420000-200, 420000-100, 
                                    420000-45, 420000-35, 420000-25, 420000-15, 420000-10, 420000-5, 
                                    420000, 420001, 420002, 420003, 420004, 420005, 420006, 420007, 420008, 420009, 420010, 420015, 420025, 420035, 420045,
                                    420100, 420200, 420300, 420400
                                    ])

        df_all = pd.DataFrame()
        for alt in alt_list: 
            trajectory = DHGridDist(planet, planet.radius + altitude, degree=density_deg)

            Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
            Call_a = Call_r0_gm.load()

            C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
            Call_a_C22 = C22_r0_gm.load()

            stats = {}
            for deg in deg_list:
                Call_r0_gm = SphericalHarmonics(model_file, degree=deg, trajectory=trajectory)
                Clm_a = Call_r0_gm.load()
                
                grid_true = Grid(trajectory=trajectory, accelerations=Call_a-Call_a_C22)
                grid_pred = Grid(trajectory=trajectory, accelerations=Clm_a-Call_a_C22)
                diff = grid_pred - grid_true

                two_sigma_mask = np.where(grid_true.total > np.mean(grid_true.total) + 2*np.std(grid_true.total))
                two_sigma_mask_compliment = np.where(grid_true.total < np.mean(grid_true.total) + 2*np.std(grid_true.total))
                two_sig_features = diff.total[two_sigma_mask]
                two_sig_features_comp = diff.total[two_sigma_mask_compliment]

                three_sigma_mask = np.where(grid_true.total > np.mean(grid_true.total) + 3*np.std(grid_true.total))
                three_sigma_mask_compliment = np.where(grid_true.total < np.mean(grid_true.total) + 3*np.std(grid_true.total))
                three_sig_features = diff.total[three_sigma_mask]
                three_sig_features_comp = diff.total[three_sigma_mask_compliment]
            
                entries = {
                        'alt' : [alt],
                        'deg_' + str(deg) + 'diff_mean' : [np.mean(diff.total)],
                        'deg_' + str(deg) + 'diff_std' : [np.std(diff.total)],
                        'deg_' + str(deg) + 'diff_median' : [np.median(diff.total)],
                        
                        'deg_' + str(deg) + 'sigma_2_mean' : [np.mean(np.sqrt(np.square(two_sig_features)))],
                        'deg_' + str(deg) + 'sigma_2_std' : [np.std(np.sqrt(np.square(two_sig_features)))],
                        'deg_' + str(deg) + 'sigma_2_median' : [np.median(np.sqrt(np.square(two_sig_features)))],

                        'deg_' + str(deg) + 'sigma_2_c_mean' : [np.mean(np.sqrt(np.square(two_sig_features_comp)))],
                        'deg_' + str(deg) + 'sigma_2_c_std' : [np.std(np.sqrt(np.square(two_sig_features_comp)))],
                        'deg_' + str(deg) + 'sigma_2_c_median' : [np.median(np.sqrt(np.square(two_sig_features_comp)))],
                    
                        'deg_' + str(deg) + 'sigma_3_mean' : [np.mean(np.sqrt(np.square(three_sig_features)))],
                        'deg_' + str(deg) + 'sigma_3_std' : [np.std(np.sqrt(np.square(three_sig_features)))],
                        'deg_' + str(deg) + 'sigma_3_median' : [np.median(np.sqrt(np.square(three_sig_features)))],

                        'deg_' + str(deg) + 'sigma_3_c_mean' : [np.mean(np.sqrt(np.square(three_sig_features_comp)))],
                        'deg_' + str(deg) + 'sigma_3_c_std' : [np.std(np.sqrt(np.square(three_sig_features_comp)))],
                        'deg_' + str(deg) + 'sigma_3_c_median' : [np.median(np.sqrt(np.square(three_sig_features_comp)))],

                        'deg_' + str(deg) + 'max_error' : [np.max(np.sqrt(np.square(diff.total)))]
                    }
                stats.update(entries)
            df = pd.DataFrame().from_dict(stats).set_index('alt')
            df_all = df_all.append(df)
        df_all.to_pickle(df_file)


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

        