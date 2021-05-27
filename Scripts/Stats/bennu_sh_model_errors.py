import numpy as np
import pandas as pd
import pickle

from GravNN.Support.Grid import Grid
from GravNN.Support.StateObject import StateObject

from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data

from GravNN.CelestialBodies.Planets import Earth
from GravNN.CelestialBodies.Asteroids import Bennu

from GravNN.Trajectories import SurfaceDist, DHGridDist, ReducedGridDist


def main():
    
    planet = Bennu()
    sh_file = planet.sh_obj_file
    density_deg = 180
    max_deg = 37

    radius_min = planet.radius
    
    # df_file = "Data/Dataframes/sh_stats_bennu_brillouin.data"
    # trajectory = DHGridDist(planet, radius_min, degree=density_deg)

    df_file = "Data/Dataframes/sh_stats_bennu_surface.data"
    trajectory = SurfaceDist(planet, planet.obj_hf_file)

    x, a , u = get_poly_data(trajectory, planet.obj_hf_file)

    state_true = StateObject(trajectory=trajectory, accelerations=a)


    deg_list =  np.linspace(1, 37, 37,dtype=int)[1:]
    df_all = pd.DataFrame()

    for deg in deg_list:

        x, a, u = get_sh_data(trajectory, sh_file, max_deg=deg, deg_removed=0)

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

        extras = {'deg' : [deg],
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
        
        df = pd.DataFrame().from_dict(entries).set_index('deg')
        df_all = df_all.append(df)

    df_all.to_pickle(df_file)


if __name__ == "__main__":
    main()