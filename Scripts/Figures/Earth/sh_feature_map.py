import os
import numpy as np

from GravNN.Support.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapBase import MapBase
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories import DHGridDist, ReducedGridDist

import matplotlib.pyplot as plt

def std_masks(grid, sigma):
    sigma_mask = np.where(grid.total > (np.mean(grid.total) + sigma*np.std(grid.total)))
    sigma_mask_compliment = np.where(grid.total < (np.mean(grid.total) + sigma*np.std(grid.total)))
    return sigma_mask, sigma_mask_compliment

def get_grid(degree, map_type='world'):
    planet = Earth()
    model_file = planet.sh_file
    density_deg = 180

    radius_min = planet.radius
    if map_type == 'world':
        trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    else:
        trajectory = ReducedGridDist(planet, radius_min, density_deg, reduction=0.25)
    Clm_r0_gm = SphericalHarmonics(model_file, degree=degree, trajectory=trajectory)
    Clm_a = Clm_r0_gm.load().accelerations
    return Grid(trajectory, Clm_a)

def feature_mask(map_type):

    grid_true = get_grid(1000, map_type)
    grid_C00 = get_grid(0, map_type)
    grid_C22 = get_grid(2, map_type)

    grid_Call_m_C22 = grid_true - grid_C22


    # Histogram on two scales
    perturbation_distribution = grid_Call_m_C22.total.reshape((-1,))
    mean = np.mean(grid_Call_m_C22.total)
    sigma = np.std(grid_Call_m_C22.total)
    outliers  = grid_Call_m_C22.total > mean + 2*sigma
    compliment = grid_Call_m_C22.total < mean + 2*sigma

    index = np.where(grid_Call_m_C22.total == np.max(grid_Call_m_C22.total))[0]

    map_visualization = MapBase()
    im = map_visualization.new_map(grid_Call_m_C22.total, cmap='coolwarm')#'binary')#, "Total")
    
    outlier_mask = np.zeros(grid_true.total.shape)
    compliment_mask = np.zeros(grid_true.total.shape)

    mask = np.ones(grid_Call_m_C22.total.shape, dtype=bool)
    mask[index] = False

    outlier_mask[outliers] = np.max(grid_Call_m_C22.total[mask])
    outlier_mask[compliment] = np.min(grid_Call_m_C22.total)

    #compliment_mask[compliment] = 1.0

    #https://stackoverflow.com/questions/10127284/overlay-imshow-plots-in-matplotlib
    map_visualization.new_map(outlier_mask, alpha=0.1, cmap='PiYG')
    #map_visualization.new_map(compliment_mask, alpha=1, cmap='autumn')




map_vis = VisualizationBase()

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12.0)

#map_vis.fig_size = (3, 1.8)

directory = os.path.abspath('.') +"/Plots/OneOff/"
os.makedirs(directory, exist_ok=True)

def main():
    map_type = 'pacific'
    map_type = 'world'
    feature_mask(map_type)

    plt.show()



if __name__ == "__main__":
    main()