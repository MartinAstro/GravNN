import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import get_sh_data
from GravNN.Support.StateObject import StateObject
from GravNN.Support.Statistics import mean_std_median, sigma_mask
from GravNN.Trajectories import FibonacciDist


def main():
    max_deg = 1000
    points = 250000

    # df_file = "Data/Dataframes/sh_stats_moon_Brillouin.data"
    # planet = Moon()
    # sh_file = planet.sh_file

    df_file = "Data/Dataframes/sh_stats_Brillouin_percent.data"
    planet = Earth()
    sh_file = planet.sh_file

    trajectory = FibonacciDist(planet, planet.radius, points)

    x, a, u = get_sh_data(trajectory, sh_file, max_deg=max_deg, deg_removed=2)
    grid_true = StateObject(trajectory=trajectory, accelerations=a)

    deg_list = np.linspace(1, 150, 150, dtype=int)[2:]
    deg_list = np.append(deg_list, [175, 200, 215, 250, 300, 400, 500, 700, 900])
    df_all = pd.DataFrame()
    for deg in deg_list:
        x, a, u = get_sh_data(trajectory, sh_file, max_deg=deg, deg_removed=2)

        grid_pred = StateObject(trajectory=trajectory, accelerations=a)
        diff = (grid_pred - grid_true) / grid_true * 100

        # This ensures the same features are being evaluated independent of what degree
        # is taken off at beginning
        sigma_1_mask, sigma_1_mask_compliment = sigma_mask(grid_true.total, 1)
        sigma_2_mask, sigma_2_mask_compliment = sigma_mask(grid_true.total, 2)
        sigma_3_mask, sigma_3_mask_compliment = sigma_mask(grid_true.total, 3)

        data = diff.total
        rse_stats = mean_std_median(data, prefix="rse")
        sigma_1_stats = mean_std_median(data, sigma_1_mask, "sigma_1")
        sigma_1_c_stats = mean_std_median(data, sigma_1_mask_compliment, "sigma_1_c")
        sigma_2_stats = mean_std_median(data, sigma_2_mask, "sigma_2")
        sigma_2_c_stats = mean_std_median(data, sigma_2_mask_compliment, "sigma_2_c")
        sigma_3_stats = mean_std_median(data, sigma_3_mask, "sigma_3")
        sigma_3_c_stats = mean_std_median(data, sigma_3_mask_compliment, "sigma_3_c")

        extras = {
            "deg": [deg],
            "max_error": [np.max(diff.total)],
        }

        entries = {
            **rse_stats,
            **sigma_1_stats,
            **sigma_1_c_stats,
            **sigma_2_stats,
            **sigma_2_c_stats,
            **sigma_3_stats,
            **sigma_3_c_stats,
            **extras,
        }
        # for d in (rse_stats, percent_stats, percent_rel_stats, sigma_2_stats,
        # sigma_2_c_stats, sigma_3_stats, sigma_3_c_stats): entries.update(d)

        df = pd.DataFrame().from_dict(entries).set_index("deg")
        df_all = df_all.append(df)

    df_all.to_pickle(df_file)
    print(df_all["rse_mean"])
    print(df_all["sigma_2_mean"])


if __name__ == "__main__":
    main()
