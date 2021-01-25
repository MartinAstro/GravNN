        
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    
    planet = Eros()
    model_file = planet.model_25k
    density_deg = 90

    radius_min = planet.radius
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)

    x, a, u = get_poly_data(trajectory, model_file)
    
    grid_true = Grid(trajectory=trajectory, accelerations=a)

    mapUnit = 'mGal'
    mapUnit = 'm/s^2'
    
    if mapUnit == 'mGal':
        scale = 10000.0
    else:
        scale = 1
    k = 100
    k_max_val = np.partition(grid_true.total.flatten(), -k)[-k]
    k_min_val = np.partition(grid_true.total.flatten(), k)[k]
    vlim = [k_min_val, k_max_val*scale]
    #vlim = [0, np.max(grid_true.total)*scale] 

    directory = os.path.abspath('.') +"/Plots/OneOff/"
    os.makedirs(directory, exist_ok=True)



    map_vis = MapVisualization(mapUnit)
    map_vis.fig_size = map_vis.full_page
    import matplotlib.pylab as pl
    from matplotlib.colors import ListedColormap


    my_cmap = 'viridis'
    #vlim = [0, 15]
    im = map_vis.new_map(grid_true.total, vlim=vlim, cmap=my_cmap)#,log_scale=True)
    map_vis.add_colorbar(im, '[mGal]', vlim, extend='max')
    #map_vis.save(plt.gcf(), directory + "sh_brillouin_map.pdf")


    plt.show()
if __name__ == "__main__":
    main()
