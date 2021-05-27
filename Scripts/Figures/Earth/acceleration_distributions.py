import os
import numpy as np
import pandas as pd
import pickle

from GravNN.Support.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories import DHGridDist, ReducedGridDist

import matplotlib.pyplot as plt

def std_masks(grid, sigma):
    sigma_mask = np.where(grid.total > (np.mean(grid.total) + sigma*np.std(grid.total)))
    sigma_mask_compliment = np.where(grid.total < (np.mean(grid.total) + sigma*np.std(grid.total)))
    return sigma_mask, sigma_mask_compliment

def get_grid(degree, map_type='world'):
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180

    radius_min = planet.radius
    if map_type == 'world':
        trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    else:
        trajectory = ReducedGridDist(planet, radius_min, density_deg, reduction=0.25)
    Clm_r0_gm = SphericalHarmonics(model_file, degree=degree, trajectory=trajectory)
    Clm_a = Clm_r0_gm.load().accelerations
    return Grid(trajectory, Clm_a)

def acceleration_histogram_masks(map_type):

    grid_true = get_grid(1000, map_type)
    grid_C00 = get_grid(0, map_type)
    grid_C22 = get_grid(2, map_type)

    grid_Call_m_C22 = grid_true - grid_C22

    # Histogram on two scales
    perturbation_distribution = grid_Call_m_C22.total.reshape((-1,))
    mean = np.mean(perturbation_distribution)
    sigma = np.std(perturbation_distribution)
    outliers  = perturbation_distribution[perturbation_distribution > mean + 4*sigma]
    compliment = perturbation_distribution[perturbation_distribution < mean + 4*sigma]

    #sets up the axis and gets histogram data
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.hist(compliment, bins=100, color='g')
    ax2.hist(outliers, bins=100, color='r')
    plt.show()

    ax1.hist([compliment, outliers], color=['g', 'r'])
    n, bins, patches = ax1.hist([compliment, outliers])
    ax1.cla() #clear the axis

    #plots the histogram data
    width = (bins[1] - bins[0]) * 0.4
    bins_shifted = bins + width
    ax1.bar(bins[:-1], n[0], width, align='edge', color=colors[0])
    ax2.bar(bins_shifted[:-1], n[1], width, align='edge', color=colors[1])

    #finishes the plot
    ax1.set_ylabel("Count", color=colors[0])
    ax2.set_ylabel("Count", color=colors[1])
    ax1.tick_params('y', colors=colors[0])
    ax2.tick_params('y', colors=colors[1])
    plt.tight_layout()
    plt.show()


def acceleration_distribution_plots(map_type):

    grid_true = get_grid(1000, map_type)
    grid_C00 = get_grid(0, map_type)
    grid_C22 = get_grid(2, map_type)
    grid_C33 = get_grid(3, map_type)
    grid_C4040 = get_grid(40, map_type)

    # Primary Figures 
    map_vis.newFig()
    plt.hist(grid_true.total.reshape((-1,)), 100)
    plt.xlabel(r"$||a||$ [m/$s^2$]")
    plt.ylabel("Frequency")
    map_vis.save(plt.gcf(), directory + "a_norm_hist.pdf")

    grid_Call_m_C00 = grid_true - grid_C00
    map_vis.newFig()
    plt.hist(grid_Call_m_C00.total.reshape((-1,)), 100)
    plt.xlabel(r"$||a-a_0||$ [m/$s^2$]")
    plt.ylabel("Frequency")

    grid_Call_m_C22 = grid_true - grid_C22
    map_vis.newFig()
    plt.hist(grid_Call_m_C22.total.reshape((-1,)), 100)
    plt.xlabel(r"$||a-(a_0 + \nabla(U_{c_{22}}))||$ [m/$s^2$]")
    plt.ylabel("Frequency")
    map_vis.save(plt.gcf(), directory + "a_norm_m_grad_C22_hist.pdf")


    # grid_Call_m_C33 = grid_true - grid_C33
    # plt.figure()
    # plt.hist(grid_Call_m_C33.total.reshape((-1,)), 100)
    # plt.xlabel(r"$||a-(a_0 + \text{grad}(U_{c_{33}}))||$ [m/$s^2$]")
    # plt.ylabel("Frequency")


    # Supplimentary Figures 

    # plt.figure()
    # plt.hist(grid_C22.total.reshape((-1,)), 100)
    # plt.xlabel(r"$||a_{C22}||$ [m/$s^2$]")
    # plt.ylabel("Frequency")

    # grid_Call_m_C4040 = grid_true - grid_C4040
    # plt.figure()
    # plt.hist(grid_Call_m_C4040.total.reshape((-1,)), 100)
    # plt.xlabel(r"$||a-(a0 + grad(U_c4040))||$ [m/$s^2$]")
    # plt.ylabel("Frequency")

def cumulative_distribution():
    map_type='world'
    grid_true = get_grid(1000, map_type)
    grid_C00 = get_grid(0, map_type)
    grid_C22 = get_grid(2, map_type)

    grid_Call_m_C22 = grid_true - grid_C22
    sorted_data = np.sort(grid_Call_m_C22.total.reshape((-1,)))

    total = len(sorted_data) - len(np.where(grid_Call_m_C22.total > (np.mean(grid_Call_m_C22.total) + 1*np.std(grid_Call_m_C22.total)))[0])
    total2 = len(sorted_data) - len(np.where(grid_Call_m_C22.total > (np.mean(grid_Call_m_C22.total) + 2*np.std(grid_Call_m_C22.total)))[0])
    total3 = len(sorted_data) - len(np.where(grid_Call_m_C22.total > (np.mean(grid_Call_m_C22.total) + 3*np.std(grid_Call_m_C22.total)))[0])

    x = np.arange(0, len(sorted_data), 1)
    y = np.zeros(x.shape)
    y0 = 0
    for i in range(len(x)):

        y0 += sorted_data[i]
        y[i] = y0
    
    plt.figure()
    plt.plot(x,y)
    plt.ylabel("Cumulative Sum")
    plt.xlabel("Data Number")

    plt.figure()
    plt.plot(x,np.gradient(y))
    plt.vlines(total, 0, np.max(np.gradient(y)), colors='r', label='1STD')
    plt.vlines(total2, 0, np.max(np.gradient(y)), colors='g', label='2STD')
    plt.vlines(total3, 0, np.max(np.gradient(y)), colors='y', label='3STD')
    plt.ylabel("Cumulative Sum Gradient")
    plt.xlabel("Data Number")

def acceleration_histogram_std():
    # Distribution Stats 
    map_type = 'world'
    grid_true = get_grid(1000, map_type)
    grid_C00 = get_grid(0, map_type)
    grid_C22 = get_grid(2, map_type)

    grid_Call_m_C22 = grid_true - grid_C22
    map_vis.newFig()
    plt.hist(grid_Call_m_C22.total.reshape((-1,)), 1000)
    # plt.vlines(np.mean(grid_Call_m_C22.total), 0, 5000, linestyle='--' label='mean')
    # plt.vlines(np.mean(grid_Call_m_C22.total) + np.std(grid_Call_m_C22.total), 0, 5000, colors='g', label='mean+std')
    # plt.vlines(np.mean(grid_Call_m_C22.total) + 2*np.std(grid_Call_m_C22.total), 0, 5000, colors='r', label='mean+2std')
    # plt.vlines(np.mean(grid_Call_m_C22.total) + 3*np.std(grid_Call_m_C22.total), 0, 5000, colors='y', label='mean+3std')
    # plt.vlines(np.mean(grid_Call_m_C22.total) + 4*np.std(grid_Call_m_C22.total), 0, 5000, colors='c', label='mean+4std')
    # plt.vlines(np.mean(grid_Call_m_C22.total) + 5*np.std(grid_Call_m_C22.total), 0, 5000, colors='m', label='mean+5std')
    plt.xlim([0, 1E-2])
    #plt.vlines(1E-8, 0, 5000, colors='y', label='SRP at Earth')
    #plt.xlabel("||a-(a0 + grad(U_c22))|| [m/s^2]")
    plt.xlabel('$|\delta \mathbf{a}|$' + '[m/s$^2$]')
    plt.ylabel('Frequency')
    map_vis.save(plt.gcf(), directory + 'earth_acc_histogram.pdf')

def acceleration_masks():
    map_type = 'world'
    grid_true = get_grid(1000, map_type)
    grid_C00 = get_grid(0, map_type)
    grid_C22 = get_grid(2, map_type)

    grid_Call_m_C22 = grid_true - grid_C22
    five_sigma_mask, five_sigma_mask_compliment = std_masks(grid_Call_m_C22, 5)
    four_sigma_mask, four_sigma_mask_compliment = std_masks(grid_Call_m_C22, 4)
    three_sigma_mask, three_sigma_mask_compliment = std_masks(grid_Call_m_C22, 3)
    two_sigma_mask, two_sigma_mask_compliment = std_masks(grid_Call_m_C22, 2)
    one_sigma_mask, one_sigma_mask_compliment = std_masks(grid_Call_m_C22, 1)

    # plt.figure()
    # plt.hist(grid_Call_m_C22.total[five_sigma_mask].reshape((-1,)), 100)
    # plt.hist(grid_Call_m_C22.total[five_sigma_mask_compliment].reshape((-1,)), 100)

    map_vis = MapVisualization(unit='m/s^2')
    map_vis.fig_size = map_vis.half_page
    map_vis.tick_interval = [60,60]

    plt.rc('text', usetex=True)

    layer_grid = grid_true - grid_C22
    layer_grid.total[one_sigma_mask_compliment] = 0
    layer_grid.total[one_sigma_mask] = 1
    layer_grid.total[two_sigma_mask] = 2
    layer_grid.total[three_sigma_mask] = 3
    layer_grid.total[four_sigma_mask] = 4
    layer_grid.total[five_sigma_mask] = 5
    fig, ax = map_vis.plot_grid(layer_grid.total, 'sigma map')#, cmap='Dark2')
    map_vis.save(fig, directory + 'earth_sigma_map.pdf')
    # vlim = [0, np.max(grid_Call_m_C22.total)*10000]
    
    # directory = os.path.abspath('.') +"/Plots/"
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "Truth", vlim)
    # #map_vis.save(fig, directory + 'OneOff/true_' + map_type + '.pdf')

    # two_sigma_values = grid_Call_m_C22.total[two_sigma_mask]
    # grid_Call_m_C22.total[two_sigma_mask] = 0.0
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "2-Sigma Feature Compliment", vlim)
    # #map_vis.save(fig, directory + 'OneOff/two_sigma_mask_compliment_' + map_type + '.pdf')
    # grid_Call_m_C22.total[two_sigma_mask] = two_sigma_values

    # two_sigma_compliment_values = grid_Call_m_C22.total[two_sigma_mask_compliment]
    # grid_Call_m_C22.total[two_sigma_mask_compliment] = 0.0
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "2-Sigma Features", vlim)
    # #map_vis.save(fig, directory + 'OneOff/two_sigma_mask_' + map_type + '.pdf')
    # grid_Call_m_C22.total[two_sigma_mask_compliment] = two_sigma_compliment_values

    # three_sigma_values = grid_Call_m_C22.total[three_sigma_mask]
    # grid_Call_m_C22.total[three_sigma_mask] = 0.0
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "3-Sigma Feature Compliment", vlim)
    # #map_vis.save(fig, directory + 'OneOff/three_sigma_mask_compliment_' + map_type + '.pdf')
    # grid_Call_m_C22.total[three_sigma_mask] = three_sigma_values

    # three_sigma_compliment_values = grid_Call_m_C22.total[three_sigma_mask_compliment]
    # grid_Call_m_C22.total[three_sigma_mask_compliment] = 0.0
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "3-Sigma Features", vlim)
    # #map_vis.save(fig, directory + 'OneOff/three_sigma_mask_' + map_type + '.pdf')
    # grid_Call_m_C22.total[three_sigma_mask_compliment] = three_sigma_compliment_values


map_vis = VisualizationBase(halt_formatting=False)
map_vis.fig_size = map_vis.half_page

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12.0)

#map_vis.fig_size = (3, 1.8)

directory = os.path.abspath('.') +"/Plots/OneOff/"
os.makedirs(directory, exist_ok=True)

def main():
    map_type = 'pacific'
    map_type = 'world'
    #acceleration_distribution_plots(map_type)
    #acceleration_histogram_masks(map_type)
    acceleration_histogram_std()
    #cumulative_distribution()
    #acceleration_masks()
   
    plt.show()



if __name__ == "__main__":
    main()