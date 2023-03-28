import numpy as np
from numba import njit


def mean_std_median(data, mask=None, prefix=None, stat_types=["mean", "std", "median"]):
    if mask is not None:
        data = data[mask]
    if prefix is not None:
        prefix = prefix + "_"
    values = {}
    values.update({prefix + "mean": [np.mean(data)]} if "mean" in stat_types else {})
    values.update({prefix + "std": [np.std(data)]} if "std" in stat_types else {})
    values.update(
        {prefix + "median": [np.median(data)]} if "median" in stat_types else {},
    )

    return values


@njit(cache=True)
def sigma_mask(grid, sigma):
    mask = np.where(grid > np.mean(grid) + sigma * np.std(grid))
    mask_compliment = np.where(grid < np.mean(grid) + sigma * np.std(grid))
    return mask, mask_compliment
