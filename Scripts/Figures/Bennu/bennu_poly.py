        
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase


def main():
    
    planet = Bennu()
    obj_file = planet.obj_hf_file
    sh_file = planet.sh_obj_file
    density_deg = 180

    trajectory = DHGridDist(planet, planet.radius, degree=density_deg)
    map_trajectory = trajectory

    poly_gm = Polyhedral(planet, obj_file, trajectory)
    acc_poly = poly_gm.load().accelerations

    max_deg = 9
    Call_r0_gm = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
    acc_sh = Call_r0_gm.load().accelerations

    max_deg = 0
    Call_r0_gm = SphericalHarmonics(sh_file, degree=max_deg, trajectory=trajectory)
    acc_sh_point_mass = Call_r0_gm.load().accelerations

    grid_true = Grid(trajectory=trajectory, accelerations=acc_poly-acc_sh_point_mass)
    grid_pred = Grid(trajectory=trajectory, accelerations=acc_sh-acc_sh_point_mass)
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
    #map_vis.save(fig, directory + "Bennu_SH_Brillouin_Diff.pdf")

    plt.show()
if __name__ == "__main__":
    main()
