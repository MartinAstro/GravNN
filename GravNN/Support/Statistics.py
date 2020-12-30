import numpy as np

def mean_std_median(data, mask=None, prefix=None):
    if mask is not None:
        data = data[mask]
    if prefix is not None:
        prefix = prefix + "_"
    values = {             
            prefix +'mean' : [np.mean(data)],
            prefix +'std' : [np.std(data)],
            prefix +'median' : [np.median(data)],
            }
    return values

def sigma_mask(grid, sigma):
    mask = np.where(grid > np.mean(grid) + sigma*np.std(grid))
    mask_compliment = np.where(grid < np.mean(grid) + sigma*np.std(grid))
    return mask, mask_compliment

    # two_sigma_mask, two_sigma_mask_compliment = sigma_mask(data, 2) 
    # three_sigma_mask, three_sigma_mask_compliment = sigma_mask(data, 3) 

    # rse_total = grid
    # two_sig_features = grid[two_sigma_mask]
    # two_sig_features_comp = grid[two_sigma_mask_compliment]
    # three_sig_features = grid[three_sigma_mask]
    # three_sig_features_comp = grid[three_sigma_mask_compliment]
    
    # rse_stats = mean_std_median(rse_total, 'rse')
    # sigma_2_stats = mean_std_median(two_sig_features, 'sigma_2')
    # sigma_2_c_stats = mean_std_median(two_sig_features_comp, 'sigma_2_c')
    # sigma_3_stats = mean_std_median(three_sig_features, 'sigma_3')
    # sigma_3_c_stats = mean_std_median(three_sig_features_comp, 'sigma_3_c')

    # entries = {
    #             **rse_stats,
    #             **sigma_2_stats,
    #             **sigma_2_c_stats,
    #             **sigma_3_stats,
    #             **sigma_3_c_stats
    #         }
    # return entries