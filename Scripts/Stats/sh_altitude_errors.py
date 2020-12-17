import numpy as np
import pandas as pd
import pickle

from GravNN.Support.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist

def main():
    
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000

    df_file = "sh_stats_altitude.data"

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


if __name__ == "__main__":
    main()