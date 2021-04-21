        
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

def format_potential_as_Nx3(model):
    Clm_p = model.potentials
    Clm_a = model.accelerations
    Clm_p_3 = np.zeros(np.shape(Clm_a))
    Clm_p_3[:,0] = Clm_p
    return Clm_p_3

def main():
    
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000

    radius_min = planet.radius
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)

    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory).load()
    Call_p = format_potential_as_Nx3(Call_r0_gm)
    
    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=trajectory).load()
    C22_p = format_potential_as_Nx3(C22_r0_gm)

    grid_true_potential = Grid(trajectory=trajectory, accelerations=Call_p-C22_p)
    grid_true_accelerations = Grid(trajectory=trajectory, accelerations=Call_r0_gm.accelerations-C22_r0_gm.accelerations)

   
    directory = os.path.abspath('.') +"/Plots/OneOff/"
    os.makedirs(directory, exist_ok=True)

    mapUnit =  "m/s^2"# 'mGal'
    map_vis = MapVisualization(mapUnit)
    map_vis.fig_size = map_vis.full_page
    map_vis.plot_grid(grid_true_potential.r, "Potential")
    map_vis.plot_grid(grid_true_accelerations.total, "Accelerations", vlim=[0,10/10000.])#, vlim=[0,10])


    plt.show()

if __name__ == "__main__":
    main()
