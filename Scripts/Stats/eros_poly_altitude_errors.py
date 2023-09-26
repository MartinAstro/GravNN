import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import get_poly_data
from GravNN.Support.Grid import Grid
from GravNN.Support.Statistics import mean_std_median, sigma_mask
from GravNN.Trajectories import DHGridDist


def main():
    # TODO: Need to make this a multi-index of ["SH_deg_2"]["diff_mean"],
    # ['28385867.273768]['diff_mean']
    planet = Eros()
    obj_file = planet.model_25k
    density_deg = 90

    df_file = "Data/Dataframes/poly_stats_altitude.data"

    alt_list = [
        0,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10000,
    ]  # Every 0.5 kilometers above surface
    alt_list = np.sort(np.unique(alt_list))
    model_list = [planet.model_12k, planet.model_6k, planet.model_3k]
    df_all = pd.DataFrame()
    for alt in alt_list:
        trajectory = DHGridDist(planet, planet.radius + alt, degree=density_deg)

        x, a, u = get_poly_data(trajectory, obj_file)
        grid_true = Grid(trajectory=trajectory, accelerations=a)

        stats = {}
        for model in model_list:
            x, a, u = get_poly_data(trajectory, model)
            grid_pred = Grid(trajectory=trajectory, accelerations=a)
            diff = grid_pred - grid_true

            sigma_2_mask, sigma_2_mask_compliment = sigma_mask(grid_true.total, 2)
            sigma_3_mask, sigma_3_mask_compliment = sigma_mask(grid_true.total, 3)

            prefix = model.split("Blender_")[1].split(".")[0] + "_"

            rse_stats = mean_std_median(diff.total, prefix=prefix + "rse")
            sigma_2_stats = mean_std_median(
                diff.total,
                sigma_2_mask,
                prefix + "sigma_2",
            )
            sigma_2_c_stats = mean_std_median(
                diff.total,
                sigma_2_mask_compliment,
                prefix + "sigma_2_c",
            )
            sigma_3_stats = mean_std_median(
                diff.total,
                sigma_3_mask,
                prefix + "sigma_3",
            )
            sigma_3_c_stats = mean_std_median(
                diff.total,
                sigma_3_mask_compliment,
                prefix + "sigma_3_c",
            )

            extras = {
                "alt": [alt],
                prefix + "_max_error": [np.max(diff.total)],
            }

            entries = {
                **rse_stats,
                **sigma_2_stats,
                **sigma_2_c_stats,
                **sigma_3_stats,
                **sigma_3_c_stats,
                **extras,
            }

            stats.update(entries)
        df = pd.DataFrame().from_dict(stats).set_index("alt")
        df_all = df_all.append(df)
    df_all.to_pickle(df_file)


if __name__ == "__main__":
    main()
