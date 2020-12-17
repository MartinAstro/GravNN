        
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sigfig
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Support.Grid import Grid
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase

map_vis = MapVisualization()

def plot_1d_curve(df, metric, y_label):
    fig, ax = map_vis.newFig(fig_size=map_vis.tri_page)
    plt.semilogx(df.index*(df.index+1), df[metric])
    plt.ylabel(y_label)
    plt.xlabel("Params, $p$")
    ax.ticklabel_format(axis='y', style='sci',scilimits=(0, 0),  useMathText=True)
    return fig

def main():
    """ 
    This generates 
    1) Spherical Harmonic RSE median curves Brillouin, LEO, and GEO 
    2) Spherical Harmonic RSE 2 sigma curves Brillouin, LEO, and GEO
    3) Spherical Harmonic RSE 2 sigma compliment curves Brillouin, LEO, and GEO
    """

    # TODO: Consider the proper figure size. 

    directory = os.path.abspath('.') +"/Plots/OneOff/SH_RSE/"
    os.makedirs(directory, exist_ok=True)

    with open("sh_stats_full_grid.data", 'rb') as f:
        brillouin_df = pickle.load(f)
    with open("sh_stats_LEO.data", 'rb') as f:
        leo_df = pickle.load(f)
    with open("sh_stats_GEO.data", 'rb') as f:
        geo_df = pickle.load(f)

    # ! RSE MEDIAN 
    # Brillouin 0 km
    fig = plot_1d_curve(brillouin_df, 'rse_median', 'Median RSE')
    map_vis.save(fig, directory + "SH_Median_RSE_Brillouin.pdf")

    # LEO 420 km 
    fig = plot_1d_curve(leo_df, 'rse_median', 'Median RSE')
    map_vis.save(fig, directory + "SH_Median_RSE_LEO.pdf")    

    # GEO 35,786 km 
    fig = plot_1d_curve(geo_df, 'rse_median', 'Median RSE')
    map_vis.save(fig, directory + "SH_Median_RSE_GEO.pdf")


    # ! RSE 2 SIGMA MEDIAN 
    # Brillouin 0 km
    fig = plot_1d_curve(brillouin_df, 'sigma_2_median', 'Median RSE')
    map_vis.save(fig, directory + "SH_sigma_2_Median_RSE_Brillouin.pdf")

    # LEO 420 km 
    fig = plot_1d_curve(leo_df, 'sigma_2_median', 'Median RSE')
    map_vis.save(fig, directory + "SH_sigma_2_Median_RSE_LEO.pdf")    

    # GEO 35,786 km 
    fig = plot_1d_curve(geo_df, 'sigma_2_median', 'Median RSE')
    map_vis.save(fig, directory + "SH_sigma_2_Median_RSE_GEO.pdf")


    # ! RSE 2 SIGMA COMPLIMENT MEDIAN 
    # Brillouin 0 km
    fig = plot_1d_curve(brillouin_df, 'sigma_2_c_median', 'Median RSE')
    map_vis.save(fig, directory + "SH_sigma_2_c_Median_RSE_Brillouin.pdf")

    # LEO 420 km 
    fig = plot_1d_curve(leo_df, 'sigma_2_c_median', 'Median RSE')
    map_vis.save(fig, directory + "SH_sigma_2_c_Median_RSE_LEO.pdf")    

    # GEO 35,786 km 
    fig = plot_1d_curve(geo_df, 'sigma_2_c_median', 'Median RSE')
    map_vis.save(fig, directory + "SH_sigma_2_c_Median_RSE_GEO.pdf")

if __name__ == "__main__":
    main()
