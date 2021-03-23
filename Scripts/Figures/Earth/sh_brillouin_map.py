        
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000

    radius_min = planet.radius
    
    df_file = "Data/Dataframes/sh_stats_Brillouin.data"
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)

    # df_file = "Data/Dataframes/sh_stats_reduced_grid.data"
    # trajectory = ReducedGridDist(planet, radius_min, degree=density_deg, reduction=0.25)
    # map_trajectory = trajectory

    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
    Call_a = Call_r0_gm.load().accelerations

    # Clm_r0_gm = SphericalHarmonics(model_file, degree=100, trajectory=trajectory)
    # Clm_a = Clm_r0_gm.load().accelerations
    
    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory)
    C22_a = C22_r0_gm.load().accelerations
        
    grid_true = Grid(trajectory=trajectory, accelerations=Call_a-C22_a)
    #grid_pred = Grid(trajectory=trajectory, accelerations=Clm_a-C22_a)
    #diff = grid_pred - grid_true
    
    k = 10
    k_max_val = np.partition(grid_true.total.flatten(), -k)[-k]
    k_min_val = np.partition(grid_true.total.flatten(), k)[k]
    vlim = [k_min_val, k_max_val*10000.0]
    vlim = [0, np.max(grid_true.total)*10000.0] 

    directory = os.path.abspath('.') +"/Plots/OneOff/"
    os.makedirs(directory, exist_ok=True)

    mapUnit = 'mGal'
    map_vis = MapVisualization(mapUnit)
    map_vis.fig_size = map_vis.full_page
    import matplotlib.pylab as pl
    from matplotlib.colors import ListedColormap

    # Choose colormap
    cmap = pl.cm.terrain

    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    #my_cmap[:,-1] = np.linspace(0.75, 0.95, cmap.N)
    midway = int(cmap.N/2)
    quarter = int(cmap.N/16)
    #my_cmap[midway:,1] /= 2
    my_cmap[:,0] += 0.1
    my_cmap[:,0] /= np.max(my_cmap[:,0])

    # my_cmap[:midway,2] /= 1.25 
    # my_cmap[midway:,2] /= 1.75 # Darker blues at bottom
    # my_cmap[quarter:,1] = my_cmap[quarter:,1]/1.05 # Less green 
    # my_cmap[quarter:,0] = my_cmap[quarter:,0]**0.25 # More red in top half
    # my_cmap[quarter:,2] = my_cmap[quarter:,2]/1.25 # More red in top half

    my_cmap[:quarter,2] = my_cmap[:quarter,2]**1.25

    my_cmap[quarter:,0] = np.linspace(my_cmap[quarter,0], 1.0, 256-quarter)
    my_cmap[quarter:,1] = np.linspace(my_cmap[quarter,1], 0.5, 256-quarter)
    my_cmap[quarter:,2] = np.linspace(my_cmap[quarter,2], 0.0, 256-quarter)

    #my_cmap[quarter:,0] = my_cmap[quarter:,0]**0.25 # More red in top half
    #my_cmap[quarter:,3] = 0.8


    # my_cmap[:,0] = np.linspace(0,0.5, cmap.N)
    # my_cmap[:,2] = np.linspace(0.5, 0, cmap.N)
    # my_cmap[:,1] = np.linspace(0.3, 0.1, cmap.N)


    # my_cmap[:,0] -= 0.1
    # my_cmap[:,0] = my_cmap[:,0]/np.max(my_cmap[:,0])

    # my_cmap[:,2] += 0.1
    #my_cmap[:,2] = my_cmap[:,2]/np.max(my_cmap[:,2])

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)


    my_cmap = 'viridis'
    vlim= [0, 40]
    im = map_vis.new_map(grid_true.total, vlim=vlim, cmap=my_cmap)#,log_scale=True)
    map_vis.add_colorbar(im, '[mGal]', vlim, extend='max')
    map_vis.save(plt.gcf(), directory + "sh_brillouin_map.pdf")


    # map_vis.fig_size = (5*4,3.5*4)
    # fig, ax = map_vis.newFig()

    # plt.subplot(311)
    # im = map_vis.new_map(grid_true.total, vlim=vlim, log_scale=False)
    # map_vis.add_colorbar(im, '[mGal]', vlim)
    
    # plt.subplot(312)
    # im = map_vis.new_map(grid_pred.total, vlim=vlim, log_scale=False)
    # map_vis.add_colorbar(im, '[mGal]', vlim)
    
    # plt.subplot(313)
    # im = map_vis.new_map(diff.total, vlim=vlim, log_scale=False)
    # map_vis.add_colorbar(im, '[mGal]', vlim)
    
    # map_vis.save(fig, directory + "sh_brillouin_diff.pdf")

    plt.show()
if __name__ == "__main__":
    main()
