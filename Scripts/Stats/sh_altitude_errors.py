import numpy as np
import pandas as pd
import pickle

from GravNN.Support.Grid import Grid
from GravNN.Support.StateObject import StateObject
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.CelestialBodies.Planets import Earth, Moon
from GravNN.CelestialBodies.Asteroids import Bennu

from GravNN.Trajectories import FibonacciDist
from GravNN.Support.Statistics import sigma_mask, mean_std_median

def get_altitude_list(planet):
    if planet == Earth():
        alt_list = np.linspace(0, 500000, 50, dtype=float) # Every 10 kilometers above surface
        window = np.array([5, 15, 45, 100, 300]) # Close to surface distribution
        alt_list = np.concatenate([alt_list, window, 420000+window, 420000-window])
        altitudes = np.sort(np.unique(alt_list))
    elif planet == Moon():
        altitudes = np.linspace(0, 55000, 55, dtype=float) # Every 1 kilometers above surface
    elif planet == Bennu():
        exit("Not implemented yet")      
    else:
        exit("Selected planet not implemented yet")

    return sh_stats_df, altitudes 

def main():
    # TODO: Need to make this a multi-index of ["SH_deg_2"]["diff_mean"], ['28385867.273768]['diff_mean']
    planet = Earth()
    df_file = "Data/Dataframes/sh_stats_earth_altitude_v2.data"
    deg_list =  np.arange(2, 350, 25)#[2, 25, 50, 55, 75, 100, 110, 150, 200, 215]
    alt_list = np.linspace(0, 500000, 50, dtype=float) # Every 10 kilometers above surface
    window = np.array([5, 15, 45, 100, 300]) # Close to surface distribution
    alt_list = np.concatenate([alt_list, window, 420000+window, 420000-window])
    alt_list = np.sort(np.unique(alt_list))

    planet = Moon()
    df_file = "Data/Dataframes/sh_stats_moon_altitude.data"
    deg_list =  np.arange(2, 275, 25)#[2, 25, 50, 55, 75, 100, 110, 150, 200, 215]
    alt_list = np.linspace(0, 50000, 50, dtype=float)#get_altitude_list(planet)
    alt_list = np.concatenate([alt_list, np.linspace(50000, 55000, 2, dtype=float)[1:]])

    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000
    points = 250000


    df_all = pd.DataFrame()
    columns = pd.MultiIndex.from_product([['rse_mean', 'sigma_1_mean', 'sigma_1_c_mean', 'sigma_2_mean', 'sigma_2_c_mean', 'sigma_3_mean', 'sigma_3_c_mean'], deg_list])
    index = pd.Index(alt_list)
    data = np.zeros((len(alt_list), len(deg_list)*7))
    for i in range(0,len(alt_list)): 
        alt = alt_list[i]
        trajectory = FibonacciDist(planet, planet.radius + alt, points=points)

        x, a, u = get_sh_data(trajectory, model_file, max_deg=max_deg, deg_removed=2)
        grid_true = StateObject(trajectory=trajectory, accelerations=a)

        parameters = np.array([])
        rse_mean_array = np.zeros((1, len(deg_list)))
        sigma_1_mean_array = np.zeros((1, len(deg_list)))
        sigma_1_c_mean_array = np.zeros((1, len(deg_list)))
        sigma_2_mean_array = np.zeros((1, len(deg_list)))
        sigma_2_c_mean_array = np.zeros((1, len(deg_list)))
        sigma_3_mean_array = np.zeros((1, len(deg_list)))
        sigma_3_c_mean_array = np.zeros((1, len(deg_list)))


        for j in range(len(deg_list)):
            deg = deg_list[j]

            x, a, u = get_sh_data(trajectory, model_file, max_deg=deg, deg_removed=2)
            grid_pred = StateObject(trajectory=trajectory, accelerations=a)
            diff = grid_pred - grid_true

            one_sigma_mask, one_sigma_mask_compliment = sigma_mask(grid_true.total, 1)
            two_sigma_mask, two_sigma_mask_compliment = sigma_mask(grid_true.total, 2)
            three_sigma_mask, three_sigma_mask_compliment = sigma_mask(grid_true.total, 3)

            rse_mean_array[0][j] = np.mean(diff.total) #mean_std_median(diff.total, prefix='rse')['rse_mean'][0]
            sigma_1_mean_array[0][j] = np.mean(diff.total[one_sigma_mask])#mean_std_median(diff.total, one_sigma_mask, "sigma_1")['sigma_1_mean'][0]
            sigma_1_c_mean_array[0][j] = np.mean(diff.total[one_sigma_mask_compliment])#mean_std_median(diff.total, one_sigma_mask_compliment, "sigma_1_c")['sigma_1_c_mean'][0]
            sigma_2_mean_array[0][j] = np.mean(diff.total[two_sigma_mask])#mean_std_median(diff.total, two_sigma_mask, "sigma_2")['sigma_2_mean'][0]
            sigma_2_c_mean_array[0][j] = np.mean(diff.total[two_sigma_mask_compliment])#mean_std_median(diff.total, two_sigma_mask_compliment, "sigma_2_c")['sigma_2_c_mean'][0]
            sigma_3_mean_array[0][j] = np.mean(diff.total[three_sigma_mask])# mean_std_median(diff.total, three_sigma_mask, "sigma_3")['sigma_3_mean'][0]
            sigma_3_c_mean_array[0][j] = np.mean(diff.total[three_sigma_mask_compliment])# mean_std_median(diff.total, three_sigma_mask_compliment, "sigma_3_c")['sigma_3_c_mean'][0]

        data[i] = np.concatenate((rse_mean_array, sigma_1_mean_array, sigma_1_c_mean_array, sigma_2_mean_array, sigma_2_c_mean_array, sigma_3_mean_array, sigma_3_c_mean_array), axis=1)

    # Now when you call df[420000] -- return a df with columns of degree
    df = pd.DataFrame(data, columns=columns, index=index)
    print(df)
    df.to_pickle(df_file)


if __name__ == "__main__":
    main()