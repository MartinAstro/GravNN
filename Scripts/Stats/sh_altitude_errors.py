import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Planets import Earth, Moon
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Support.StateObject import StateObject
from GravNN.Support.Statistics import sigma_mask
from GravNN.Trajectories import FibonacciDist


def main():
    """
    Generate the error @ a fixed altitude + fixed degree.
    """
    planet = Earth()
    df_file = "Data/Dataframes/sh_stats_earth_altitude_v2.data"
    deg_list = np.arange(2, 350, 25)
    alt_list = np.linspace(0, 500000, 50, dtype=float)
    window = np.array([5, 15, 45, 100, 300])  # Close to surface distribution
    alt_list = np.concatenate([alt_list, window, 420000 + window, 420000 - window])
    alt_list = np.sort(np.unique(alt_list))

    planet = Moon()
    df_file = "Data/Dataframes/sh_stats_moon_altitude.data"
    deg_list = np.arange(2, 275, 25)
    alt_list = np.linspace(0, 50000, 50, dtype=float)
    alt_list = np.concatenate([alt_list, np.linspace(50000, 55000, 2, dtype=float)[1:]])

    sh_file = planet.sh_file
    max_deg = 1000
    points = 250000

    metrics = [
        "rse_mean",
        "sigma_1_mean",
        "sigma_1_c_mean",
        "sigma_2_mean",
        "sigma_2_c_mean",
        "sigma_3_mean",
        "sigma_3_c_mean",
    ]
    columns = pd.MultiIndex.from_product([metrics, deg_list])
    index = pd.Index(alt_list)
    data = np.zeros((len(alt_list), len(deg_list) * len(metrics)))
    for i, alt in enumerate(alt_list):
        trajectory = FibonacciDist(planet, planet.radius + alt, points=points)

        x, a, u = get_sh_data(trajectory, sh_file, max_deg=max_deg, deg_removed=2)
        grid_true = StateObject(trajectory=trajectory, accelerations=a)

        rse_mean_array = np.zeros((1, len(deg_list)))
        sigma_1_f_mean_array = np.zeros((1, len(deg_list)))
        sigma_1_c_mean_array = np.zeros((1, len(deg_list)))
        sigma_2_f_mean_array = np.zeros((1, len(deg_list)))
        sigma_2_c_mean_array = np.zeros((1, len(deg_list)))
        sigma_3_f_mean_array = np.zeros((1, len(deg_list)))
        sigma_3_c_mean_array = np.zeros((1, len(deg_list)))

        for j in range(len(deg_list)):
            deg = deg_list[j]

            x, a, u = get_sh_data(trajectory, sh_file, max_deg=deg, deg_removed=2)
            grid_pred = StateObject(trajectory=trajectory, accelerations=a)
            diff = grid_pred - grid_true

            sigma_1_mask, sigma_1_mask_compliment = sigma_mask(grid_true.total, 1)
            sigma_2_mask, sigma_2_mask_compliment = sigma_mask(grid_true.total, 2)
            sigma_3_mask, sigma_3_mask_compliment = sigma_mask(grid_true.total, 3)

            rse_mean_array[0][j] = np.mean(diff.total)
            sigma_1_f_mean_array[0][j] = np.mean(diff.total[sigma_1_mask])
            sigma_1_c_mean_array[0][j] = np.mean(diff.total[sigma_1_mask_compliment])
            sigma_2_f_mean_array[0][j] = np.mean(diff.total[sigma_2_mask])
            sigma_2_c_mean_array[0][j] = np.mean(diff.total[sigma_2_mask_compliment])
            sigma_3_f_mean_array[0][j] = np.mean(diff.total[sigma_3_mask])
            sigma_3_c_mean_array[0][j] = np.mean(diff.total[sigma_3_mask_compliment])

        data[i] = np.concatenate(
            (
                rse_mean_array,
                sigma_1_f_mean_array,
                sigma_1_c_mean_array,
                sigma_2_f_mean_array,
                sigma_2_c_mean_array,
                sigma_3_f_mean_array,
                sigma_3_c_mean_array,
            ),
            axis=1,
        )

    # Now when you call df[420000] -- return a df with columns of degree
    df = pd.DataFrame(data, columns=columns, index=index)
    print(df)
    df.to_pickle(df_file)


if __name__ == "__main__":
    main()
