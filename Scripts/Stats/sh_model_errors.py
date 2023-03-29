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
    # model_file = planet.sh_file

    df_file = "Data/Dataframes/sh_stats_Brillouin_deg_n1.data"
    deg_removed = -1
    planet = Earth()
    model_file = planet.sh_file

    trajectory = FibonacciDist(planet, planet.radius, points)

    # df_file = "Data/Dataframes/sh_stats_DH_Brillouin.data"
    # trajectory = DHGridDist(planet,  planet.radius, degree=density_deg)

    # df_file = "Data/Dataframes/sh_stats_LEO.data"
    # trajectory = DHGridDist(planet, planet.radius + 420000.0, degree=density_deg)

    # df_file = "Data/Dataframes/sh_stats_GEO.data"
    # trajectory = DHGridDist(planet, planet.radius + 35786000.0, degree=density_deg)

    x, acc_sh, u = get_sh_data(
        trajectory,
        model_file,
        max_deg=max_deg,
        deg_removed=deg_removed,
    )
    grid_true = StateObject(trajectory=trajectory, accelerations=acc_sh)

    deg_list = np.linspace(1, 150, 150, dtype=int)[1:]
    deg_list = np.append(deg_list, [175, 200, 215, 250, 300, 400, 500, 700, 900])
    df_all = pd.DataFrame()
    for deg in deg_list:
        x, acc_sh, u = get_sh_data(trajectory, model_file, max_deg=deg, deg_removed=-1)

        grid_pred = StateObject(trajectory=trajectory, accelerations=acc_sh)
        diff = grid_pred - grid_true

        # This ensures the same features are being evaluated independent of what degree
        # is taken off at beginning
        sigma_1_mask, sigma_1_mask_comp = sigma_mask(grid_true.total, 1)
        sigma_2_mask, sigma_2_mask_comp = sigma_mask(grid_true.total, 2)
        sigma_3_mask, sigma_3_mask_comp = sigma_mask(grid_true.total, 3)

        rse_stats = mean_std_median(diff.total, prefix="rse")
        sigma_1_stats = mean_std_median(diff.total, sigma_1_mask, "sigma_1")
        sigma_1_c_stats = mean_std_median(diff.total, sigma_1_mask_comp, "sigma_1_c")
        sigma_2_stats = mean_std_median(diff.total, sigma_2_mask, "sigma_2")
        sigma_2_c_stats = mean_std_median(diff.total, sigma_2_mask_comp, "sigma_2_c")
        sigma_3_stats = mean_std_median(diff.total, sigma_3_mask, "sigma_3")
        sigma_3_c_stats = mean_std_median(diff.total, sigma_3_mask_comp, "sigma_3_c")

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

        df = pd.DataFrame().from_dict(entries).set_index("deg")
        df_all = df_all.append(df)

    df_all.to_pickle(df_file)


if __name__ == "__main__":
    main()
