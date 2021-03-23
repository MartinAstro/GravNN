import numpy as np
import pandas as pd
import pickle

from GravNN.Support.Grid import Grid
from GravNN.Support.StateObject import StateObject

from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data

from GravNN.CelestialBodies.Planets import Earth
from GravNN.CelestialBodies.Asteroids import Bennu, Eros

from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.SurfaceDist import SurfaceDist

from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Statistics import mean_std_median, sigma_mask

def main():
    """
    Generates the error of a lower fidelity polyhedral model at the surface and brillouin sphere
    """
    
    planet = Eros()
    radius_min = planet.radius
    
    df_file = "Data/Dataframes/poly_stats_eros_surface.data"
    trajectory = SurfaceDist(planet, planet.model_25k)
        
    df_file = "Data/Dataframes/poly_stats_eros_brillouin.data"
    trajectory = DHGridDist(planet, radius_min, degree=90)

    x, a ,u = get_poly_data(trajectory, planet.model_25k)
    state_true = StateObject(trajectory=trajectory, accelerations=a)

    poly_files = [planet.model_25k, planet.model_12k, planet.model_6k, planet.model_3k]
    df_all = pd.DataFrame()

    for poly_file in poly_files:
        Call_r0_gm = Polyhedral(trajectory.celestial_body, poly_file, trajectory=trajectory)
        x = trajectory.positions
        a = Call_r0_gm.load().accelerations

        state_pred = StateObject(trajectory=trajectory, accelerations=a)
        
        diff = state_pred - state_true

        # This ensures the same features are being evaluated independent of what degree is taken off at beginning
        two_sigma_mask, two_sigma_mask_compliment = sigma_mask(state_true.total, 2)
        three_sigma_mask, three_sigma_mask_compliment = sigma_mask(state_true.total, 3)

        rse_stats = mean_std_median(diff.total, prefix='rse')
        sigma_2_stats = mean_std_median(diff.total, two_sigma_mask, "sigma_2")
        sigma_2_c_stats = mean_std_median(diff.total, two_sigma_mask_compliment, "sigma_2_c")
        sigma_3_stats = mean_std_median(diff.total, three_sigma_mask, "sigma_3")
        sigma_3_c_stats = mean_std_median(diff.total, three_sigma_mask_compliment, "sigma_3_c")

        params = Call_r0_gm.mesh.vertices.shape[0]* Call_r0_gm.mesh.vertices.shape[1] + Call_r0_gm.mesh.faces.shape[0]
        extras = {'params' : [params],
                    'max_error' : [np.max(diff.total)]
                    }

        entries = { **rse_stats,
                    **sigma_2_stats,
                    **sigma_2_c_stats,
                    **sigma_3_stats,
                    **sigma_3_c_stats,
                    **extras
                    }
        #for d in (rse_stats, percent_stats, percent_rel_stats, sigma_2_stats, sigma_2_c_stats, sigma_3_stats, sigma_3_c_stats): entries.update(d)
        
        df = pd.DataFrame().from_dict(entries).set_index('params')
        df_all = df_all.append(df)

    df_all.to_pickle(df_file)


if __name__ == "__main__":
    main()