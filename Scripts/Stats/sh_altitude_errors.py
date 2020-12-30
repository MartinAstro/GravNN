import numpy as np
import pandas as pd
import pickle

from GravNN.Support.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Statistics import sigma_mask, mean_std_median

def main():
    # TODO: Need to make this a multi-index of ["SH_deg_2"]["diff_mean"], ['28385867.273768]['diff_mean']
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000

    df_file = "sh_stats_altitude.data"

    deg_list =  [2, 25, 50, 75, 100, 150, 200]
    alt_list = np.linspace(0, 500000, 100, dtype=float) # Every 0.5 kilometers above surface
    window = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 25, 35, 45, 100, 200, 300, 400]) # Close to surface distribution
    LEO_window_upper = window + 420000 # Window around LEO
    LEO_window_lower = -1.0*window + 420000
    alt_list = np.concatenate([alt_list, window, LEO_window_lower, LEO_window_upper])
    alt_list = np.sort(np.unique(alt_list))
    df_all = pd.DataFrame()
    for alt in alt_list: 
        trajectory = DHGridDist(planet, planet.radius + alt, degree=density_deg)

        x, a, u = get_sh_data(trajectory, model_file, max_deg=max_deg, deg_removed=2)
        grid_true = Grid(trajectory=trajectory, accelerations=a)

        stats = {}
        for deg in deg_list:
            x, a, u = get_sh_data(trajectory, model_file, max_deg=deg, deg_removed=2)
            grid_pred = Grid(trajectory=trajectory, accelerations=a)
            diff = grid_pred - grid_true

            two_sigma_mask, two_sigma_mask_compliment = sigma_mask(grid_true.total, 2)
            three_sigma_mask, three_sigma_mask_compliment = sigma_mask(grid_true.total, 3)

            prefix = 'deg_'+str(deg)+"_"

            rse_stats = mean_std_median(diff.total, prefix=prefix+'rse')
            sigma_2_stats = mean_std_median(diff.total, two_sigma_mask, prefix+"sigma_2")
            sigma_2_c_stats = mean_std_median(diff.total, two_sigma_mask_compliment, prefix+"sigma_2_c")
            sigma_3_stats = mean_std_median(diff.total, three_sigma_mask, prefix+"sigma_3")
            sigma_3_c_stats = mean_std_median(diff.total, three_sigma_mask_compliment, prefix+"sigma_3_c")
        
            extras = {
                    'alt' : [alt],
                     prefix + '_max_error' : [np.max(diff.total)]
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
    df_all.to_pickle(df_file)


if __name__ == "__main__":
    main()