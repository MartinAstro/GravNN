
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
    statistic = 'diff_median'

    df_file = 'N_1000000_study.data'
    nn_df = pd.read_pickle(df_file)
    ids = nn_df['id'].values

    vis = VisualizationBase()
    fig, ax = vis.newFig()

    df_file = 'sh_stats_Brillouin.data'
    sh_df = pd.read_pickle(df_file)

    ax.scatter(nn_df['params'], nn_df["Brillouin_" + statistic])
    ax.plot(sh_df.index*(sh_df.index+1), sh_df['rse_median'])
    plt.xlabel("Parameters")
    plt.ylabel("RSE")

    plt.show()
if __name__ == '__main__':
    main()
