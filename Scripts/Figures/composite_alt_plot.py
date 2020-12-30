
import os
import copy
import pickle
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks import utils
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Plotting import Plotting
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase

np.random.seed(1234)
tf.random.set_seed(0)

def main():
    planet = Earth()
    statistic = 'rse_mean'

    #df_file = 'N_1000000_study.data'
    df_file = 'N_1000000_exp_norm_study.data'
    df = pd.read_pickle(df_file)
    ids = df['id'].values

    # Generate individual curves
    figure_list = []
    for id_value in ids:
        tf.keras.backend.clear_session()

        model_id = id_value
        config, model = load_config_and_model(model_id, df_file)

        plotter = Plotting(model, config)
        fig = plotter.plot_alt_curve(statistic)
        figure_list.append(fig)
    

    # Plot composite curve
    vis = VisualizationBase()
    fig, ax = vis.newFig()

    df_file = 'sh_stats_altitude.data'
    sh_df = pd.read_pickle(df_file)

    ax.plot(sh_df.index/1000.0, sh_df['deg_2_'+statistic], linestyle='--')
    ax.plot(sh_df.index/1000.0, sh_df['deg_50_'+statistic], linestyle='--')
    ax.plot(sh_df.index/1000.0, sh_df['deg_75_'+statistic], linestyle='--')
    ax.plot(sh_df.index/1000.0, sh_df['deg_100_'+statistic], linestyle='--')

    for i in range(0,len(figure_list)):
        cur_fig = plt.figure(figure_list[i].number)
        cur_ax = cur_fig.get_axes()[0]
        data = cur_ax.get_lines()[0].get_xydata()
        ax.plot(data[:,0], data[:,1])

    plt.xlabel('Altitude [km]')
    plt.ylabel('RSE')
    plt.show()
if __name__ == '__main__':
    main()
