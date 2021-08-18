
import os
import copy
import pickle
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
#import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks import utils
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Trajectories import *
from GravNN.Support.transformations import cart2sph

np.random.seed(1234)
tf.random.set_seed(0)



def main():
    trajectories = []
    names = []
    planet = Earth()

    # # Random Brillouin
    # RandomDist.RandomDist()

    df_file = 'Data/Dataframes/sh_stats_altitude.data'
    sh_df = pd.read_pickle(df_file)

    # df_file = 'Data/Dataframes/N_1000000_exp_norm_study.data'
    # df = pd.read_pickle(df_file)   


    statistic = 'rse_mean'

    # Plot composite curve
    vis = VisualizationBase(save_directory=os.path.abspath('.') +"/Plots/")
    vis.fig_size = vis.half_page
    fig, ax2 = vis.newFig()

    line1 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_2_'+statistic], linestyle='--', label='$d=2$')
    line2 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_25_'+statistic], linestyle='--', label='$d=25$')
    line3 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_50_'+statistic], linestyle='--', label='$d=50$')
    line4 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_75_'+statistic], linestyle='--', label='$d=75$')
    line5 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_100_'+statistic], linestyle='--', label='$d=100$')
    legend1 = ax2.legend(loc='lower left')
    ax2.add_artist(legend1)
    ax2.set_ylabel('MSE')
    ax2.set_xlabel("Altitude [m]")
    vis.save(plt.gcf(), "OneOff/" + statistic + "_altitude.pdf")



    vis.fig_size = vis.tri_page

    fig, ax2 = vis.newFig()
    line1 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_2_'+'rse_mean'], label=r'RSE($\mathcal{A}$)')
    line1 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_2_'+'sigma_2_mean'], label=r'RSE($\mathcal{F}$)')
    line2 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_2_'+'sigma_2_c_mean'], label=r'RSE($\mathcal{C}$)')
    # legend1 = ax2.legend(loc='upper right')
    # ax2.add_artist(legend1)
    ax2.set_ylabel('MSE')
    ax2.set_xlabel("Altitude [m]")
    vis.save(plt.gcf(), "OneOff/" + "deg_2" + "_altitude.pdf")

    fig, ax2 = vis.newFig()
    line1 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_25_'+'rse_mean'], label=r'RSE($\mathcal{A}$)')
    line1 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_25_'+'sigma_2_mean'], label=r'RSE($\mathcal{F}$)')
    line2 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_25_'+'sigma_2_c_mean'], label=r'RSE($\mathcal{C}$)')
    # legend1 = ax2.legend(loc='upper right')
    # ax2.add_artist(legend1)
    ax2.set_ylabel('MSE')
    ax2.set_xlabel("Altitude [m]")
    vis.save(plt.gcf(), "OneOff/" + "deg_25" + "_altitude.pdf")


    fig, ax2 = vis.newFig()
    line1 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_75_'+'rse_mean'], label=r'RSE($\mathcal{A}$)')
    line1 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_75_'+'sigma_2_mean'], label=r'RSE($\mathcal{F}$)')
    line2 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_75_'+'sigma_2_c_mean'], label=r'RSE($\mathcal{C}$)')
    # legend1 = ax2.legend(loc='upper right')
    # ax2.add_artist(legend1)
    ax2.set_ylabel('MSE')
    ax2.set_xlabel("Altitude [m]")
    vis.save(plt.gcf(), "OneOff/" + "deg_75" + "_altitude.pdf")

    plt.show()
if __name__ == '__main__':
    main()
