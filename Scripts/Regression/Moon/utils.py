from GravNN.Support.Statistics import mean_std_median, sigma_mask
import numpy as np
def compute_stats(grid_true, grid_pred):
    #* Difference and stats
    diff = grid_pred - grid_true

    # This ensures the same features are being evaluated independent of what degree is taken off at beginning
    two_sigma_mask, two_sigma_mask_compliment = sigma_mask(grid_true.total, 2)
    three_sigma_mask, three_sigma_mask_compliment = sigma_mask(grid_true.total, 3)

    rse_stats = mean_std_median(diff.total, prefix='rse')
    sigma_2_stats = mean_std_median(diff.total, two_sigma_mask, "sigma_2")
    sigma_2_c_stats = mean_std_median(diff.total, two_sigma_mask_compliment, "sigma_2_c")
    sigma_3_stats = mean_std_median(diff.total, three_sigma_mask, "sigma_3")
    sigma_3_c_stats = mean_std_median(diff.total, three_sigma_mask_compliment, "sigma_3_c")

    extras = {
                'max_error' : [np.max(diff.total)]
                }

    entries = { **rse_stats,
                **sigma_2_stats,
                **sigma_2_c_stats,
                **sigma_3_stats,
                **sigma_3_c_stats,
                **extras
                    }
    return entries