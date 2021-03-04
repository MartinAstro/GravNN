
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
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks import utils
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Plotting import Plotting
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Trajectories.RandomAsteroidDist import RandomAsteroidDist
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Trajectories import *
from GravNN.Support.transformations import cart2sph


from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


np.random.seed(1234)
tf.random.set_seed(0)

mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['blue', 'green', 'red', 'orange', 'gold',  'pink',  'lime', 'salmon', 'magenta','lavender', 'yellow', 'black', 'lightblue','darkgreen', 'pink', 'brown',  'teal', 'coral',  'turquoise',  'tan', 'gold'])

def main():
    trajectories = []
    names = []
    planet = Eros()
    points = 100000
    radius_bounds = [planet.radius, planet.radius + 10000.0]

    df_file = 'Data/Dataframes/poly_stats_altitude.data'
    poly_df = pd.read_pickle(df_file)

    # df_file = 'Data/Dataframes/N_1000000_exp_norm_study.data'
    # df_exp = pd.read_pickle(df_file)   

    df_file = 'Data/Dataframes/N_100000_rand_eros_study_v2.data'
    rse_bounds = [8E-5, 3E-4]
    sigma_bounds = [2E-4, 2E-3]
    PINN = False

    # df_file = 'Data/Dataframes/N_100000_rand_eros_PINN_study_v2.data'
    # rse_bounds = [8E-5, 3E-4]
    # sigma_bounds = [2E-4, 2E-3]
    # PINN= True 

    df = pd.read_pickle(df_file)

    # Plot composite curve
    vis = VisualizationBase(save_directory=os.path.abspath('.') +"/Plots/")
    #vis.fig_size = vis.half_page

    def generate_plot(trajectory, name, statistic, legend=False, bounds=None, xlim=10000.0/1000.0):
        # Generate trajectory histogram
        positions = cart2sph(trajectory.positions)

        main_fig, ax = vis.newFig()
        plt.hist((positions[:,0] - planet.radius)/1000.0,bins=100, alpha=0.3, label=name)
        plt.xlabel('Altitude [km]')
        plt.ylabel('Frequency')
        if xlim is not None:
            plt.xlim([None, xlim])

        ax2 = ax.twinx()

        # Find all networks that trained on that trajectory
        sub_df = df[df['distribution'] == trajectory.__class__]
        sub_df = sub_df[(sub_df['params'] == 45923) | 
                        (sub_df['params'] == 11763) | 
                        (sub_df['params'] == 3083) |
                        (sub_df['params'] == 45760) | 
                        (sub_df['params'] == 11680) | 
                        (sub_df['params'] == 3040)].sort_values(by='params', ascending=False)
        #sub_df = sub_df['layers'].isin(some_values)

        try:
            sub_df = sub_df[sub_df['invert'] == trajectory.invert].sort_values(by='params', ascending=False)
        except:
            pass

        try: 
            sub_df = sub_df[sub_df['scale_parameter'] == trajectory.scale_parameter].sort_values(by='params', ascending=False)
        except:
            pass

        try: 
            sub_df = sub_df[sub_df['mu'] == trajectory.mu].sort_values(by='params', ascending=False)
        except:
            pass

        try: 
            sub_df = sub_df[sub_df['sigma'] == trajectory.sigma].sort_values(by='params', ascending=False)
        except:
            pass

        ids = sub_df['id'].values
        fig_list = []
        labels = []
        #print(sub_df['params'])

        # Generate their altitude rse plot
        for model_id in ids:
            tf.keras.backend.clear_session()

            config, model = load_config_and_model(model_id, df_file)

            plotter = Plotting(model, config)

            fig = plotter.plot_alt_curve(statistic)
            fig_list.append(fig)
            labels.append(str(df[df['id']==model_id]['layers'].values[0][3]))

        line1 = ax2.semilogy(poly_df.index/1000.0, poly_df['3k_poly_'+statistic], linestyle='--', label='$p=30,720$')
        #line2 = ax2.semilogy(poly_df.index, poly_df['deg_25_'+statistic], linestyle='--', label='$p=25$')
        line3 = ax2.semilogy(poly_df.index/1000.0, poly_df['6k_poly_'+statistic], linestyle='--', label='$p=61,440$')
        #line4 = ax2.semilogy(poly_df.index, poly_df['deg_75_'+statistic], linestyle='--', label='$p=75$')
        line5 = ax2.semilogy(poly_df.index/1000.0, poly_df['12k_poly_'+statistic], linestyle='--', label='$p=122,880$')

        if legend: 
            legend1 = ax2.legend(loc='lower left')
            ax2.add_artist(legend1)

        handles= []
        for j in range(0,len(fig_list)):
            cur_fig = plt.figure(fig_list[j].number)
            cur_ax = cur_fig.get_axes()[0]
            data = cur_ax.get_lines()[0].get_xydata()
            line, = ax2.semilogy(data[:,0], data[:,1], label='$N=' +str(labels[j]) + "$")
            handles.append(line)
            plt.close(cur_fig)

        plt.figure(main_fig.number)
        ax2.set_ylabel('MRSE')
        if legend:
            ax2.legend(handles=handles,loc='upper left')
        
        ax2.set_ylim(bottom=1E-6, top=bounds[1])
        # a = plt.axes([.3, .25, .2, .2])
        # line1 = a.semilogy(poly_df.index/1000.0, poly_df['3k_poly_'+statistic], linestyle='--', label='$d=2$')
        # #line2 = a.semilogy(poly_df.index, poly_df['deg_25_'+statistic], linestyle='--', label='$d=25$')
        # line3 = a.semilogy(poly_df.index/1000.0, poly_df['6k_poly_'+statistic], linestyle='--', label='$d=55$')
        # #line4 = a.semilogy(poly_df.index, poly_df['deg_75_'+statistic], linestyle='--', label='$d=75$')
        # line5 = a.semilogy(poly_df.index/1000.0, poly_df['12k_poly_'+statistic], linestyle='--', label='$d=110$')
        # for j in range(0,len(fig_list)):
        #     cur_fig = plt.figure(fig_list[j].number)
        #     cur_ax = cur_fig.get_axes()[0]
        #     data = cur_ax.get_lines()[0].get_xydata()
        #     line, = a.semilogy(data[:,0], data[:,1], label='$N=' +str(labels[j]) + "$")
        #     handles.append(line)
        #     plt.close(cur_fig)
        # a.set_xlim(left=0, right=10000/1000.0)
        # a.set_ylim(bottom=bounds[0], top=bounds[1])
        

        # plt.tick_params(
        #     axis='y',          # changes apply to the x-axis
        #     which='minor',      # both major and minor ticks are affected
        #     left=False,      # ticks along the left edge are off
        #     right=True,         # ticks along the top edge are off
        #     labelleft=False,
        #     labelright=True) # labels along the left edge are off
        # plt.tick_params(
        #     axis='y',          # changes apply to the x-axis
        #     which='major',      # both major and minor ticks are affected
        #     left=False,      # ticks along the left edge are off
        #     right=True,         # ticks along the top edge are off
        #     labelleft=False,
        #     labelright=False) # labels along the left edge are off

        # # a.axes.xaxis.set_ticklabels([])
        # # a.axes.yaxis.set_ticklabels([])

        # ax2.indicate_inset_zoom(a)
        if PINN:
            vis.save(plt.gcf(), "OneOff/" + name + "_PINN_distribution.pdf")
        else:
            vis.save(plt.gcf(), "OneOff/" + name + "_distribution.pdf")

    vis.fig_size = vis.half_page
    statistic = 'rse_mean'
    # * Random Brillouin
    #vis.fig_size = vis.half_page
    radius_bounds = [0, planet.radius+10000.0]
    traj = RandomAsteroidDist.RandomAsteroidDist(planet, planet.model_25k, radius_bounds, points)
    generate_plot(traj, "Random_Asteroid", statistic,legend=True, bounds=rse_bounds)

    # * Random Brillouin Features
    radius_bounds = [0, planet.radius+10000.0]
    traj = RandomAsteroidDist.RandomAsteroidDist(planet, planet.model_25k, radius_bounds, points)
    generate_plot(traj, "Random_Asteroid_sigma", 'sigma_2_mean',legend=True, bounds=sigma_bounds)



    # vis.fig_size = vis.half_page
    # # # * Exponential Narrow Average
    # invert = False
    # scale_parameter = 420000.0/10.0
    # traj = ExponentialDist.ExponentialDist(planet, radius_bounds, points, scale_parameter=[scale_parameter], invert=[invert])
    # generate_plot(traj, "Exp_Brillouin_Narrow", statistic, bounds=rse_bounds)

    # invert = True
    # scale_parameter = 420000.0/10.0
    # traj = ExponentialDist.ExponentialDist(planet, radius_bounds, points, scale_parameter=[scale_parameter], invert=[invert])
    # generate_plot(traj, "Exp_LEO_Narrow", statistic, bounds=rse_bounds)



    # # * Exponential Narrow Features
    # statistic = 'sigma_2_mean'
    # invert = False
    # scale_parameter = 420000.0/10.0
    # traj = ExponentialDist.ExponentialDist(planet, radius_bounds, points, scale_parameter=[scale_parameter], invert=[invert])
    # generate_plot(traj, "Exp_Brillouin_Narrow_sigma", statistic, bounds=sigma_bounds)

    # invert = True
    # scale_parameter = 420000.0/10.0
    # traj = ExponentialDist.ExponentialDist(planet, radius_bounds, points, scale_parameter=[scale_parameter], invert=[invert])
    # generate_plot(traj, "Exp_LEO_Narrow_sigma", statistic, bounds=sigma_bounds)



    plt.show()

   

if __name__ == '__main__':
    main()
