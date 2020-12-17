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

    radius_min = planet.radius
    
    ###########################
    #### Trajectories #########
    ###########################
    # df_file = "sh_stats_full_grid.data"
    # trajectory = DHGridDist(planet, radius_min, degree=density_deg)

    # df_file = "sh_stats_reduced_grid.data"
    # trajectory = ReducedGridDist(planet, radius_min, degree=density_deg, reduction=0.25)

    # df_file = "sh_stats_world_R100.data"
    # trajectory = DHGridDist(planet, radius_min+100, degree=density_deg)

    # df_file = "sh_stats_world_R1000.data"
    # trajectory = DHGridDist(planet, radius_min+1000, degree=density_deg)

    # df_file = "sh_stats_world_R10000.data"
    # trajectory = DHGridDist(planet, radius_min+10000, degree=density_deg)


    df_file = "sh_stats_LEO.data"
    trajectory = DHGridDist(planet, planet.radius + 420000.0, degree=density_deg)

    df_file = "sh_stats_GEO.data"
    trajectory = DHGridDist(planet, planet.radius + 35786000.0, degree=density_deg)



    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
    Call_a = Call_r0_gm.load()

    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
    Call_a_C22 = C22_r0_gm.load()

    deg_list =  np.linspace(1, 100, 100,dtype=int)[1:]
    deg_list = np.append(deg_list, [150, 200, 250, 300, 400, 500, 700, 900])
    df_all = pd.DataFrame()
    for deg in deg_list:

        Call_r0_gm = SphericalHarmonics(model_file, degree=deg, trajectory=trajectory)
        Clm_a = Call_r0_gm.load()
        
        error_Call = np.abs(np.divide((Clm_a - Call_a),Call_a))*100 # Percent Error for each component
        error_Call_m_C22 = np.abs(np.divide((Clm_a - Call_a),Call_a-Call_a_C22))*100 # Percent Error for each component

        RSE_Call = np.sqrt(np.square(Clm_a - Call_a))
        RSE_norm_Call = np.linalg.norm(RSE_Call, axis=1)
        
        grid_true = Grid(trajectory=trajectory, accelerations=Call_a-Call_a_C22)
        grid_pred = Grid(trajectory=trajectory, accelerations=Clm_a-Call_a_C22)
        diff = grid_pred - grid_true

        two_sigma_mask = np.where(grid_true.total > np.mean(grid_true.total) + 2*np.std(grid_true.total))
        two_sigma_mask_compliment = np.where(grid_true.total < np.mean(grid_true.total) + 2*np.std(grid_true.total))
        two_sig_features = diff.total[two_sigma_mask]
        two_sig_features_comp = diff.total[two_sigma_mask_compliment]

        three_sigma_mask = np.where(grid_true.total > np.mean(grid_true.total) + 3*np.std(grid_true.total))
        three_sigma_mask_compliment = np.where(grid_true.total <np.mean(grid_true.total) + 3*np.std(grid_true.total))
        three_sig_features = diff.total[three_sigma_mask]
        three_sig_features_comp = diff.total[three_sigma_mask_compliment]
       
        entries = {
                'deg' : [deg],

                'percent_mean' : [np.mean(error_Call)],
                'percent_std' : [np.std(error_Call)], 
                'percent_median' : [np.median(error_Call)],
                'percent_a0_mean' : [np.mean(error_Call[:,0])], 
                'percent_a1_mean' : [np.mean(error_Call[:,1])], 
                'percent_a2_mean' : [np.mean(error_Call[:,2])], 

                'rse_mean' : [np.mean(RSE_norm_Call)],
                'rse_std' : [np.std(RSE_norm_Call)],
                'rse_median' : [np.median(RSE_norm_Call)],
                'rse_a0_mean' : [np.mean(RSE_Call[:,0])],
                'rse_a1_mean' : [np.mean(RSE_Call[:,1])],
                'rse_a2_mean' : [np.mean(RSE_Call[:,2])],

                'percent_rel_mean' : [np.mean(error_Call_m_C22)],
                'percent_rel_std' : [np.std(error_Call_m_C22)], 
                'percent_rel_median' : [np.median(error_Call_m_C22)],
                'percent_rel_a0_mean' : [np.mean(error_Call_m_C22[:,0])], 
                'percent_rel_a1_mean' : [np.mean(error_Call_m_C22[:,1])], 
                'percent_rel_a2_mean' : [np.mean(error_Call_m_C22[:,2])],

                'sigma_3_mean' : [np.mean(np.sqrt(np.square(three_sig_features)))],
                'sigma_3_std' : [np.std(np.sqrt(np.square(three_sig_features)))],
                'sigma_3_median' : [np.median(np.sqrt(np.square(three_sig_features)))],

                'sigma_3_c_mean' : [np.mean(np.sqrt(np.square(three_sig_features_comp)))],
                'sigma_3_c_std' : [np.std(np.sqrt(np.square(three_sig_features_comp)))],
                'sigma_3_c_median' : [np.median(np.sqrt(np.square(three_sig_features_comp)))],

                'sigma_2_mean' : [np.mean(np.sqrt(np.square(two_sig_features)))],
                'sigma_2_std' : [np.std(np.sqrt(np.square(two_sig_features)))],
                'sigma_2_median' : [np.median(np.sqrt(np.square(two_sig_features)))],

                'sigma_2_c_mean' : [np.mean(np.sqrt(np.square(two_sig_features_comp)))],
                'sigma_2_c_std' : [np.std(np.sqrt(np.square(two_sig_features_comp)))],
                'sigma_2_c_median' : [np.median(np.sqrt(np.square(two_sig_features_comp)))],

                'max_error' : [np.max(np.sqrt(np.square(diff.total)))]
            }
        df = pd.DataFrame().from_dict(entries).set_index('deg')
        df_all = df_all.append(df)
    
    df_all.to_pickle(df_file)


if __name__ == "__main__":
    main()