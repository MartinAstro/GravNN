import numpy as np
import pandas as pd
import pickle

from GravNN.Support.Grid import Grid
from GravNN.Support.StateObject import StateObject

from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.FibonacciDist import FibonacciDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Statistics import sigma_mask, mean_std_median

def main():
    # TODO: Need to make this a multi-index of ["SH_deg_2"]["diff_mean"], ['28385867.273768]['diff_mean']
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000
    points = 250000
    df_file = "Data/Dataframes/sh_stats_earth_altitude.data"

    deg_list =  np.arange(2, 250, 25)#[2, 25, 50, 55, 75, 100, 110, 150, 200, 215]
    alt_list = np.linspace(0, 500000, 50, dtype=float) # Every 10 kilometers above surface
    window = np.array([5, 15, 45, 100, 300]) # Close to surface distribution
    LEO_window_upper = window + 420000 # Window around LEO
    LEO_window_lower = -1.0*window + 420000
    alt_list = np.concatenate([alt_list, window, LEO_window_lower, LEO_window_upper])
    alt_list = np.sort(np.unique(alt_list))

    df_all = pd.DataFrame()
    columns = pd.MultiIndex.from_product([['rse_mean', 'sigma_2_mean', 'sigma_2_c_mean', 'sigma_3_mean', 'sigma_3_c_mean'], deg_list])
    index = pd.Index(alt_list)
    data = np.zeros((len(alt_list), len(deg_list)*5))
    for i in range(0,len(alt_list)): 
        alt = alt_list[i]
        trajectory = FibonacciDist(planet, planet.radius + alt, points=points)

        x, a, u = get_sh_data(trajectory, model_file, max_deg=max_deg, deg_removed=2)
        grid_true = StateObject(trajectory=trajectory, accelerations=a)

        parameters = np.array([])
        rse_mean_array = np.zeros((1, len(deg_list)))
        sigma_2_mean_array = np.zeros((1, len(deg_list)))
        sigma_2_c_mean_array = np.zeros((1, len(deg_list)))
        sigma_3_mean_array = np.zeros((1, len(deg_list)))
        sigma_3_c_mean_array = np.zeros((1, len(deg_list)))


        for j in range(len(deg_list)):
            deg = deg_list[j]

            x, a, u = get_sh_data(trajectory, model_file, max_deg=deg, deg_removed=2)
            grid_pred = StateObject(trajectory=trajectory, accelerations=a)
            diff = grid_pred - grid_true

            two_sigma_mask, two_sigma_mask_compliment = sigma_mask(grid_true.total, 2)
            three_sigma_mask, three_sigma_mask_compliment = sigma_mask(grid_true.total, 3)

            rse_mean_array[0][j] = np.mean(diff.total) #mean_std_median(diff.total, prefix='rse')['rse_mean'][0]
            sigma_2_mean_array[0][j] = np.mean(diff.total[two_sigma_mask])#mean_std_median(diff.total, two_sigma_mask, "sigma_2")['sigma_2_mean'][0]
            sigma_2_c_mean_array[0][j] = np.mean(diff.total[two_sigma_mask_compliment])#mean_std_median(diff.total, two_sigma_mask_compliment, "sigma_2_c")['sigma_2_c_mean'][0]
            sigma_3_mean_array[0][j] = np.mean(diff.total[three_sigma_mask])# mean_std_median(diff.total, three_sigma_mask, "sigma_3")['sigma_3_mean'][0]
            sigma_3_c_mean_array[0][j] = np.mean(diff.total[three_sigma_mask_compliment])# mean_std_median(diff.total, three_sigma_mask_compliment, "sigma_3_c")['sigma_3_c_mean'][0]

        data[i] = np.concatenate((rse_mean_array, sigma_2_mean_array, sigma_2_c_mean_array, sigma_3_mean_array, sigma_3_c_mean_array), axis=1)

    # Now when you call df[420000] -- return a df with columns of degree
    df = pd.DataFrame(data, columns=columns, index=index)
    print(df)
    df.to_pickle(df_file)


if __name__ == "__main__":
    main()