
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
#import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks import utils
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.FigureSupport import sh_pareto_curve, nn_pareto_curve

np.random.seed(1234)
tf.random.set_seed(0)

def journal_compactness():
    planet = Earth()
    vis = VisualizationBase(save_directory=os.path.abspath('.') +"/Plots/OneOff/")
    fig, ax = vis.newFig(fig_size=vis.full_page)

    # Journal Figures
    sh_pareto_curve('Data/Dataframes/sh_stats_Brillouin_percent.data', max_deg=None, metric='median')
    plt.legend()
    vis.save(fig, "Brill_Params.pdf")

    # ! Neural Network Results
    nn_pareto_curve('Data/Dataframes/traditional_nn_df_percent.data', radius_max=planet.radius+420000, orbit_name='Brillouin_percent', linestyle='--',  metric='median')
    vis.save(fig, "NN_Brill_Params_Percent.pdf")

    nn_pareto_curve('Data/Dataframes/pinn_df_percent.data', radius_max=planet.radius+420000, orbit_name='Brillouin_percent',  marker='o',  metric='median')
    vis.save(fig, "NN_Brill_PINN_Params_Percent.pdf")


def main():
    #conference_compactness()
    journal_compactness()    
    plt.show()

if __name__ == '__main__':
    main()
