
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

    vis = VisualizationBase(save_directory=os.path.abspath('.') +"/Plots/OneOff/")

    # ! RSE MEAN Full Size
    # Brillouin 0 km    
    fig, ax = vis.newFig(fig_size=vis.full_page)

    def sh_pareto_curve(file_name, max_deg=None):
        if max_deg is not None:
            sh_df = pd.read_pickle(file_name).loc[:max_deg]
        else:
            sh_df = pd.read_pickle(file_name)

        plt.semilogx(sh_df.index*(sh_df.index+1), sh_df['rse_mean'], label=r'MRSE($\mathcal{A}$)')
        plt.semilogx(sh_df.index*(sh_df.index+1), sh_df['sigma_2_mean'], label=r'MRSE($\mathcal{F}$)')
        plt.semilogx(sh_df.index*(sh_df.index+1), sh_df['sigma_2_c_mean'], label=r'MRSE($\mathcal{C}$)')

        plt.ylabel('Mean RSE')
        plt.xlabel("Parameters")
        
        ax.ticklabel_format(axis='y', style='sci',scilimits=(0, 0),  useMathText=True)

    def nn_pareto_curve(file_name, orbit_name, linestyle=None, marker=None):
        nn_df = pd.read_pickle(file_name)
        sub_df = nn_df[nn_df['radius_max'] == planet.radius + 420000.0].sort_values(by='params')
        plt.gca().set_prop_cycle(None)
        plt.semilogx(sub_df['params'], sub_df[orbit_name+'_rse_mean'], linestyle=linestyle, marker=marker)
        plt.semilogx(sub_df['params'], sub_df[orbit_name+'_sigma_2_mean'], linestyle=linestyle, marker=marker)
        plt.semilogx(sub_df['params'], sub_df[orbit_name+'_sigma_2_c_mean'], linestyle=linestyle, marker=marker)
        plt.legend()

    sh_pareto_curve('Data/Dataframes/sh_stats_DH_Brillouin.data', max_deg=None)
    plt.legend()
    vis.save(fig, "Brill_Params.pdf")

    # ! Neural Network Results
    nn_pareto_curve('Data/Dataframes/N_1000000_Rand_Study.data', orbit_name='Brillouin', linestyle='--')
    vis.save(fig, "NN_Brill_Params.pdf")

    # ! PINN Neural Network Results
    nn_pareto_curve('Data/Dataframes/N_1000000_PINN_study.data', orbit_name='Brillouin', marker='o')
    vis.save(fig, "NN_Brill_PINN_Params.pdf")

    # # ! Optimized PINN Neural Network Results
    # nn_pareto_curve('Data/Dataframes/N_1000000_PINN_study_opt.data', orbit_name='Brillouin', marker='v')
    # vis.save(fig, "NN_Brill_PINN_Opt_Params.pdf")


    # JUST NN Representations
    # fig, ax = vis.newFig(fig_size=vis.full_page)

    # sh_pareto_curve('Data/Dataframes/sh_regress_stats_Brillouin.data')

    # # ! Neural Network Results
    # nn_pareto_curve('Data/Dataframes/N_10000_rand_study.data', orbit_name='Brillouin', linestyle='--')

    # # ! PINN Neural Network Results
    # nn_pareto_curve('Data/Dataframes/N_10000_rand_PINN_study.data', orbit_name='Brillouin', marker='o')
    # vis.save(fig, "NN_Brill_Traditional_v_PINN.pdf")


    # * LEO Orbit
    # sh_pareto_curve('Data/Dataframes/sh_stats_LEO.data')

    # # ! Neural Network Results
    # nn_pareto_curve('Data/Dataframes/N_1000000_Rand_Study.data', orbit_name='LEO', linestyle='--')
    # vis.save(fig, "NN_LEO_Params.pdf")

    # # ! PINN Neural Network Results
    # nn_pareto_curve('Data/Dataframes/N_1000000_PINN_study.data', orbit_name='LEO', marker='o')
    # vis.save(fig, "NN_LEO_PINN_Params.pdf")

    # # ! Optimized PINN Neural Network Results
    # nn_pareto_curve('Data/Dataframes/N_1000000_PINN_study_opt.data', orbit_name='LEO', marker='v')
    # vis.save(fig, "NN_LEO_PINN_Opt_Params.pdf")




    
    plt.show()
if __name__ == '__main__':
    main()
