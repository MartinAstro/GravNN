
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
import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks import utils
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Plotting import Plotting
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Trajectories import *
from GravNN.Support.transformations import cart2sph

np.random.seed(1234)
tf.random.set_seed(0)

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'orange', 'purple',  'cyan',  'lime', 'salmon', 'magenta','lavender', 'yellow', 'black', 'lightblue','darkgreen', 'pink', 'brown',  'teal', 'coral',  'turquoise',  'tan', 'gold'])

def main():
    trajectories = []
    names = []
    planet = Earth()
    points = 1000000
    radius_bounds = [planet.radius, planet.radius + 420000.0]

    # Exponential LEO
    invert = False
    scale_parameter = 420000.0/3.0
    traj = ExponentialDist.ExponentialDist(planet, radius_bounds, points, scale_parameter=[scale_parameter], invert=[invert])
    trajectories.append(traj)
    names.append('Exp Prime 1')

    scale_parameter = 420000.0/10.0
    traj = ExponentialDist.ExponentialDist(planet, radius_bounds, points, scale_parameter=[scale_parameter], invert=[invert])
    trajectories.append(traj)
    names.append('Exp Prime 2')


    # Exponential LEO
    invert = True
    scale_parameter = 420000.0/3.0
    traj = ExponentialDist.ExponentialDist(planet, radius_bounds, points, scale_parameter=[scale_parameter], invert=[invert])
    trajectories.append(traj)
    names.append('Exp Prime 1')

    scale_parameter = 420000.0/10.0
    traj = ExponentialDist.ExponentialDist(planet, radius_bounds, points, scale_parameter=[scale_parameter], invert=[invert])
    trajectories.append(traj)
    names.append('Exp Prime 2')

    # Gaussian
    radius_bounds = [planet.radius, np.inf]
    mu = planet.radius + 420000.0
    sigma = 420000.0/3.0
    traj = GaussianDist.GaussianDist(planet, radius_bounds, points, mu=[mu], sigma=[sigma])
    trajectories.append(traj)
    names.append('Normal 1')


    sigma = 420000.0/10.0
    traj = GaussianDist.GaussianDist(planet, radius_bounds, points, mu=[mu], sigma=[sigma])
    trajectories.append(traj)
    names.append('Normal 2')


    # # Random Brillouin
    # RandomDist.RandomDist()

    df_file = 'sh_stats_altitude.data'
    sh_df = pd.read_pickle(df_file)

    df_file = 'N_1000000_exp_norm_study.data'
    df = pd.read_pickle(df_file)   


    statistic = 'rse_median'

    # Plot composite curve
    vis = VisualizationBase()

    for i in range(len(trajectories)):
        
        # Generate trajectory histogram
        positions = cart2sph(trajectories[i].positions)

        main_fig, ax = vis.newFig()
        plt.hist(positions[:,0],bins=100, alpha=0.3, label=names[i])
        plt.xlabel('Altitude [km]')
        plt.ylabel('Frequency')
        plt.xlim([planet.radius, planet.radius+420000*2.0])

        ax2 = ax.twinx()

        # Find all networks that trained on that trajectory
        sub_df = df[df['distribution'] == trajectories[i].__class__]
        try:
            sub_df = sub_df[sub_df['invert'] == trajectories[i].invert].sort_values(by='params')
        except:
            pass

        try: 
            sub_df = sub_df[sub_df['scale_parameter'] == trajectories[i].scale_parameter].sort_values(by='params')
        except:
            pass

        try: 
            sub_df = sub_df[sub_df['mu'] == trajectories[i].mu].sort_values(by='params')
        except:
            pass

        try: 
            sub_df = sub_df[sub_df['sigma'] == trajectories[i].sigma].sort_values(by='params')
        except:
            pass

        ids = sub_df['id'].values
        fig_list = []
        labels = []

        # Generate their altitude rse plot
        for model_id in ids:
            tf.keras.backend.clear_session()

            config, model = load_config_and_model(model_id, df_file)
            
            plotter = Plotting(model, config)

            fig = plotter.plot_alt_curve(statistic)
            fig_list.append(fig)
            labels.append(str(df[df['id']==model_id]['layers'].values[0][3]))
        
        line1 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_2_'+statistic], linestyle='--', label='$d=2$')
        line2 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_25_'+statistic], linestyle='--', label='$d=25$')
        line3 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_50_'+statistic], linestyle='--', label='$d=50$')
        line4 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_75_'+statistic], linestyle='--', label='$d=75$')
        line5 = ax2.semilogy(sh_df.index+planet.radius, sh_df['deg_100_'+statistic], linestyle='--', label='$d=100$')
        legend1 = ax2.legend(loc='lower right')
        ax2.add_artist(legend1)
        
        handles= []
        for j in range(0,len(fig_list)):
            cur_fig = plt.figure(fig_list[j].number)
            cur_ax = cur_fig.get_axes()[0]
            data = cur_ax.get_lines()[0].get_xydata()
            line, = ax2.semilogy(data[:,0]*1000.0+planet.radius, data[:,1], label=labels[j])
            handles.append(line)
            plt.close(cur_fig)

        plt.figure(main_fig.number)
        ax2.set_ylabel('RSE')
        ax2.legend(handles=handles,loc='upper right')
        vis.save(plt.gcf(), "OneOff/" + str(trajectories[i].__class__.__name__) + "_distribution.pdf")
    plt.show()
if __name__ == '__main__':
    main()
