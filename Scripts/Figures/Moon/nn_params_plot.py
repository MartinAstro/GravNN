
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
from GravNN.CelestialBodies.Planets import Moon
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks import utils
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.FigureSupport import sh_pareto_curve, nn_pareto_curve

np.random.seed(1234)
tf.random.set_seed(0)

def journal_compactness():
    # TODO: Need to generate network dataframes 
    planet = Moon()
    vis = VisualizationBase(save_directory=os.path.abspath('.') +"/Plots/Moon/")
    fig, ax = vis.newFig(fig_size=vis.full_page)

    sh_pareto_curve('Data/Dataframes/sh_stats_moon_Brillouin.data', max_deg=None, sigma=2)
    plt.legend()
    vis.save(fig, "Brill_Params.pdf")

    # ! Neural Network Results
    nn_pareto_curve('Data/Dataframes/moon_traditional_nn_df.data', radius_max=planet.radius + 50000, orbit_name='Brillouin', linestyle='--',sigma=2)
    vis.save(fig, "NN_Brill_Params.pdf")

    # ! PINN Neural Network Results
    nn_pareto_curve('Data/Dataframes/moon_pinn_df.data', radius_max=planet.radius + 50000, orbit_name='Brillouin', marker='o',sigma=2)
    vis.save(fig, "NN_Brill_PINN_Params.pdf")


def main():
    journal_compactness()  
    plt.show()

    
if __name__ == '__main__':
    main()
