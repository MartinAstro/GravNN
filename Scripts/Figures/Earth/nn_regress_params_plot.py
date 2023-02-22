
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
from GravNN.Visualization.MapBase import MapBase
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.FigureSupport import nn_pareto_curve, sh_pareto_curve

np.random.seed(1234)
tf.random.set_seed(0)

def conference_regression():
    planet = Earth()
    vis = VisualizationBase(save_directory=os.path.abspath('.') +"/Plots/OneOff/")
    fig, ax = vis.newFig(fig_size=vis.full_page)

    sh_pareto_curve('Data/Dataframes/sh_regress_stats_33_Brillouin.data')
    #sh_pareto_curve('Data/Dataframes/sh_regress_stats_33_Random.data')

    # ! Neural Network Results
    nn_pareto_curve('Data/Dataframes/N_10000_rand_study.data', radius_max=planet.radius + 420000, orbit_name='Brillouin', linestyle='--')
    vis.save(fig, "NN_Regress_Brill_Params.pdf")

    # ! PINN Neural Network Results
    nn_pareto_curve('Data/Dataframes/N_10000_rand_PINN_study.data', radius_max=planet.radius + 420000, orbit_name='Brillouin', marker='o')
    vis.save(fig, "NN_Regress_Brill_PINN_Params.pdf")

def main():
    conference_regression()
    plt.show()

if __name__ == '__main__':
    main()
