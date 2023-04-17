import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Support.StateObject import StateObject
from GravNN.Support.Statistics import mean_std_median, sigma_mask
from GravNN.Trajectories import DHGridDist, SurfaceDist


def main():
    """
    Generates the error of a lower fidelity polyhedral model at the surface and
    brillouin sphere
    """

    planet = Eros()
    radius_min = planet.radius

    df_file = "Data/Dataframes/poly_stats_eros_surface.data"
    trajectory = SurfaceDist(planet, planet.model_25k)

    df_file = "Data/Dataframes/poly_stats_eros_brillouin.data"
    trajectory = DHGridDist(planet, radius_min, degree=90)

    x, a, u = get_poly_data(trajectory, planet.model_25k)
    state_true = StateObject(trajectory=trajectory, accelerations=a)

    poly_files = [planet.model_25k, planet.model_12k, planet.model_6k, planet.model_3k]
    df_all = pd.DataFrame()

    for poly_file in poly_files:
        Call_r0_gm = Polyhedral(
            trajectory.celestial_body,
            poly_file,
            trajectory=trajectory,
        )
        a = Call_r0_gm.load().accelerations

        state_pred = StateObject(trajectory=trajectory, accelerations=a)

        diff = state_pred - state_true

        # This ensures the same features are being evaluated independent of what degree
        #  is taken off at beginning
        sigma_2_mask, sigma_2_mask_compliment = sigma_mask(state_true.total, 2)
        sigma_3_mask, sigma_3_mask_compliment = sigma_mask(state_true.total, 3)

        data = diff.total
        rse_stats = mean_std_median(data, prefix="rse")
        sigma_2_stats = mean_std_median(data, sigma_2_mask, "sigma_2")
        sigma_2_c_stats = mean_std_median(data, sigma_2_mask_compliment, "sigma_2_c")
        sigma_3_stats = mean_std_median(data, sigma_3_mask, "sigma_3")
        sigma_3_c_stats = mean_std_median(data, sigma_3_mask_compliment, "sigma_3_c")

        params = (
            Call_r0_gm.mesh.vertices.shape[0] * Call_r0_gm.mesh.vertices.shape[1]
            + Call_r0_gm.mesh.faces.shape[0]
        )
        extras = {
            "params": [params],
            "max_error": [np.max(data)],
        }

        entries = {
            **rse_stats,
            **sigma_2_stats,
            **sigma_2_c_stats,
            **sigma_3_stats,
            **sigma_3_c_stats,
            **extras,
        }
        # for d in (rse_stats, percent_stats, percent_rel_stats, sigma_2_stats,
        # sigma_2_c_stats, sigma_3_stats, sigma_3_c_stats): entries.update(d)

        df = pd.DataFrame().from_dict(entries).set_index("params")
        df_all = df_all.append(df)

    df_all.to_pickle(df_file)


if __name__ == "__main__":
    main()
