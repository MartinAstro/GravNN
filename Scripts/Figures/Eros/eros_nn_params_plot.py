
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
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks import utils
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Plotting import Plotting
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.FigureSupport import nn_pareto_curve, sh_pareto_curve

np.random.seed(1234)
tf.random.set_seed(0)

def conference_compactness():
    planet = Eros()
    vis = VisualizationBase(save_directory=os.path.abspath('.') +"/Plots/OneOff/")
    fig, ax = vis.newFig(fig_size=vis.full_page)

    # ! RSE MEAN Full Size
    sh_pareto_curve('Data/Dataframes/poly_stats_eros_brillouin.data', max_deg=None)
    vis.save(fig, "Eros_Brill_Params.pdf")

    # ! Neural Network Results
    nn_pareto_curve('Data/Dataframes/N_100000_rand_eros_study_v2.data', radius_max=planet.radius + 420000, orbit_name='Brillouin', linestyle='--')
    vis.save(fig, "NN_Eros_Brill_Params.pdf")

    # ! PINN Neural Network Results
    nn_pareto_curve('Data/Dataframes/N_100000_rand_eros_PINN_study_v2.data', radius_max=planet.radius + 420000, orbit_name='Brillouin', marker='o')
    vis.save(fig, "NN_Eros_Brill_PINN_Params.pdf")

    # # * Surface
    # fig, ax = vis.newFig(fig_size=vis.full_page)
    # sh_pareto_curve('poly_stats_eros_surface.data', max_deg=None)
    # vis.save(fig, "Eros_Surface_Params.pdf")

    # # ! Neural Network Results
    # nn_pareto_curve('N_100000_rand_eros_study_v2.data', orbit_name='Surface', linestyle='--')
    # vis.save(fig, "NN_Eros_Surface_Params.pdf")

    # # ! PINN Neural Network Results
    # nn_pareto_curve('N_100000_rand_eros_PINN_study_v2.data', orbit_name='Surface', marker='o')
    # vis.save(fig, "NN_Eros_Surface_PINN_Params.pdf")


def main():
    conference_compactness()   
    plt.show()


if __name__ == '__main__':
    main()
