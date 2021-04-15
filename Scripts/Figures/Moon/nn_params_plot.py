
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
from GravNN.Networks.Plotting import Plotting
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase

np.random.seed(1234)
tf.random.set_seed(0)

def journal_compactness():
    # TODO: Need to generate network dataframes 
    planet = Moon()
    vis = VisualizationBase(save_directory=os.path.abspath('.') +"/Plots/Moon/")
    fig, ax = vis.newFig(fig_size=vis.full_page)

    sh_pareto_curve('Data/Dataframes/sh_stats_moon_Brillouin.data', max_deg=None)
    plt.legend()
    vis.save(fig, "Brill_Params.pdf")

    # # ! Neural Network Results
    # nn_pareto_curve('Data/Dataframes/N_1000000_Rand_Study.data', radius_max=planet.radius + 50000, orbit_name='Brillouin', linestyle='--')
    # vis.save(fig, "NN_Brill_Params.pdf")

    # # ! PINN Neural Network Results
    # nn_pareto_curve('Data/Dataframes/N_1000000_PINN_study.data', radius_max=planet.radius + 50000, orbit_name='Brillouin', marker='o')
    # vis.save(fig, "NN_Brill_PINN_Params.pdf")

    # # ! Optimized PINN Neural Network Results
    # nn_pareto_curve('Data/Dataframes/N_1000000_PINN_study_opt.data', radius_max=planet.radius + 50000, orbit_name='Brillouin', marker='v')
    # vis.save(fig, "NN_Brill_PINN_Opt_Params.pdf")

def main():
    journal_compactness()  
    plt.show()

    
if __name__ == '__main__':
    main()
