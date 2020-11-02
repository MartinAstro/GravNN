        
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000

    radius_min = planet.radius
    
    df_file = "sh_stats_full_grid.data"
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)

    # df_file = "sh_stats_reduced_grid.data"
    trajectory = ReducedGridDist(planet, radius_min, degree=density_deg, reduction=0.25)
    map_trajectory = trajectory

    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
    Call_a = Call_r0_gm.load()

    Clm_r0_gm = SphericalHarmonics(model_file, degree=100, trajectory=trajectory)
    Clm_a = Clm_r0_gm.load()
    
    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=map_trajectory)
    C22_a = C22_r0_gm.load()
        
    grid_true = Grid(trajectory=trajectory, accelerations=Call_a-C22_a)
    grid_pred = Grid(trajectory=trajectory, accelerations=Clm_a-C22_a)
    diff = grid_pred - grid_true
    
    mapUnit = 'mGal'
    map_vis = MapVisualization(mapUnit)
    plt.rc('text', usetex=False)
    map_vis.fig_size = (5*4,3.5*4)
    fig, ax = map_vis.newFig()
    vlim = [0, np.max(grid_true.total)*10000.0] 
    plt.subplot(311)
    im = map_vis.new_map(grid_true.total, vlim=vlim, log_scale=False)
    map_vis.add_colorbar(im, '[mGal]', vlim)
    
    plt.subplot(312)
    im = map_vis.new_map(grid_pred.total, vlim=vlim, log_scale=False)
    map_vis.add_colorbar(im, '[mGal]', vlim)
    
    plt.subplot(313)
    im = map_vis.new_map(diff.total, vlim=vlim, log_scale=False)
    map_vis.add_colorbar(im, '[mGal]', vlim)
    
    directory = os.path.abspath('.') +"/Plots/OneOff/"
    os.makedirs(directory, exist_ok=True)
    map_vis.save(fig, directory + "Example_SH_Diff_Map.pdf")

if __name__ == "__main__":
    main()
