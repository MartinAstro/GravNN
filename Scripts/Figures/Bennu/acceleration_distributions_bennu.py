import os
import numpy as np
import pandas as pd
import pickle

from GravNN.Support.StateObject import StateObject
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapBase import MapBase
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics

from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.Trajectories import DHGridDist, SurfaceDist

import matplotlib.pyplot as plt

def std_masks(stateObject, sigma):
    sigma_mask = np.where(stateObject.total > (np.mean(stateObject.total) + sigma*np.std(stateObject.total)))
    sigma_mask_compliment = np.where(stateObject.total < (np.mean(stateObject.total) + sigma*np.std(stateObject.total)))
    return sigma_mask, sigma_mask_compliment

def get_poly_state_object(map_type='sphere'):
    planet = Bennu()
    obj_file = planet.obj_file
    density_deg = 180

    radius_min = planet.radius
    if map_type == 'sphere':
        trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    else:
        trajectory = SurfaceDist(planet, obj_file)
    Clm_r0_gm = Polyhedral(planet, obj_file, trajectory)
    Clm_a = Clm_r0_gm.load().accelerations

    return StateObject(trajectory, Clm_a)

def get_sh_state_object_C00(map_type='sphere'):
    planet = Bennu()
    obj_file = planet.obj_file
    density_deg = 180

    radius_min = planet.radius
    if map_type == 'sphere':
        trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    else:
        trajectory = SurfaceDist(planet, obj_file)
    Clm_r0_gm = SphericalHarmonics(planet.sh_obj_file, 0, trajectory)
    Clm_a = Clm_r0_gm.load().accelerations
    return StateObject(trajectory, Clm_a)

def acceleration_distribution_plots(map_type):

    state_object_true = get_poly_state_object(map_type)
    state_object_C00 = get_sh_state_object_C00(map_type)

    # Primary Figures 
    plt.figure()
    plt.hist(state_object_true.total.reshape((-1,)), 100)
    plt.xlabel("||a|| [m/s^2]")
    plt.ylabel("Frequency")
    #map_vis.save(plt.gcf(), directory + "a_norm_hist.pdf")

    state_object_Call_m_C00 = state_object_true - state_object_C00
    plt.figure()
    plt.hist(state_object_Call_m_C00.total.reshape((-1,)), 100)
    plt.xlabel("||a-a0|| [m/s^2]")
    plt.ylabel("Frequency")
    #map_vis.save(plt.gcf(), directory + "a_m_C00_norm_hist.pdf")

def cumulative_distribution(map_type):
    state_object_true = get_poly_state_object(map_type)
    state_object_C00 = get_sh_state_object_C00(map_type)

    state_object_Call_m_C00 = state_object_true - state_object_C00
    sorted_data = np.sort(state_object_Call_m_C00.total.reshape((-1,)))

    total = len(sorted_data) - len(np.where(state_object_Call_m_C00.total > (np.mean(state_object_Call_m_C00.total) + 1*np.std(state_object_Call_m_C00.total)))[0])
    total2 = len(sorted_data) - len(np.where(state_object_Call_m_C00.total > (np.mean(state_object_Call_m_C00.total) + 2*np.std(state_object_Call_m_C00.total)))[0])
    total3 = len(sorted_data) - len(np.where(state_object_Call_m_C00.total > (np.mean(state_object_Call_m_C00.total) + 3*np.std(state_object_Call_m_C00.total)))[0])

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

def acceleration_histogram_std(map_type):
    # Distribution Stats 
    
    state_object_true = get_poly_state_object(map_type)
    state_object_C00 = get_sh_state_object_C00(map_type)

    state_object_Call_m_C00 = state_object_true - state_object_C00
    plt.figure()
    plt.hist(state_object_Call_m_C00.total.reshape((-1,)), 1000)
    plt.vlines(np.mean(state_object_Call_m_C00.total), 0, 5000, colors='r', label='mean')
    plt.vlines(np.mean(state_object_Call_m_C00.total) + np.std(state_object_Call_m_C00.total), 0, 5000, colors='g', label='mean+std')
    plt.vlines(np.mean(state_object_Call_m_C00.total) + 2*np.std(state_object_Call_m_C00.total), 0, 5000, colors='y', label='mean+2std')
    plt.vlines(1E-8, 0, 5000, colors='y', label='SRP at Earth')
    plt.xlabel("||a-a0|| [m/s^2]")

def acceleration_masks(map_type):
    state_object_true = get_poly_state_object(map_type)
    state_object_C00 = get_sh_state_object_C00(map_type)

    state_object_Call_m_C00 = state_object_true - state_object_C00
    five_sigma_mask, five_sigma_mask_compliment = std_masks(state_object_Call_m_C00, 5)
    four_sigma_mask, four_sigma_mask_compliment = std_masks(state_object_Call_m_C00, 4)
    three_sigma_mask, three_sigma_mask_compliment = std_masks(state_object_Call_m_C00, 3)
    two_sigma_mask, two_sigma_mask_compliment = std_masks(state_object_Call_m_C00, 2)
    one_sigma_mask, one_sigma_mask_compliment = std_masks(state_object_Call_m_C00, 1)

    # plt.figure()
    # plt.hist(state_object_Call_m_C00.total[five_sigma_mask].reshape((-1,)), 100)
    # plt.hist(state_object_Call_m_C00.total[five_sigma_mask_compliment].reshape((-1,)), 100)

    map_vis = MapBase(unit='mGal')
    plt.rc('text', usetex=False)

    layer_stateObject = state_object_true - state_object_C00
    layer_stateObject.total[one_sigma_mask_compliment] = 0.0
    layer_stateObject.total[one_sigma_mask] = 1.0
    layer_stateObject.total[two_sigma_mask] = 2.0
    layer_stateObject.total[three_sigma_mask] = 3.0
    layer_stateObject.total[four_sigma_mask] = 4.0
    layer_stateObject.total[five_sigma_mask] = 5.0
    map_vis.plot_stateObject(layer_stateObject.total, 'sigma map')

    vlim = [0, np.max(state_object_Call_m_C00.total)*10000]
    
    directory = os.path.abspath('.') +"/Plots/"
    fig, ax = map_vis.plot_stateObject(state_object_Call_m_C00.total, "Truth", vlim)
    #map_vis.save(fig, directory + 'OneOff/true_' + map_type + '.pdf')

    two_sigma_values = state_object_Call_m_C00.total[two_sigma_mask]
    state_object_Call_m_C00.total[two_sigma_mask] = 0.0
    fig, ax = map_vis.plot_stateObject(state_object_Call_m_C00.total, "2-Sigma Feature Compliment", vlim)
    #map_vis.save(fig, directory + 'OneOff/two_sigma_mask_compliment_' + map_type + '.pdf')
    state_object_Call_m_C00.total[two_sigma_mask] = two_sigma_values

    two_sigma_compliment_values = state_object_Call_m_C00.total[two_sigma_mask_compliment]
    state_object_Call_m_C00.total[two_sigma_mask_compliment] = 0.0
    fig, ax = map_vis.plot_stateObject(state_object_Call_m_C00.total, "2-Sigma Features", vlim)
    #map_vis.save(fig, directory + 'OneOff/two_sigma_mask_' + map_type + '.pdf')
    state_object_Call_m_C00.total[two_sigma_mask_compliment] = two_sigma_compliment_values

    three_sigma_values = state_object_Call_m_C00.total[three_sigma_mask]
    state_object_Call_m_C00.total[three_sigma_mask] = 0.0
    fig, ax = map_vis.plot_stateObject(state_object_Call_m_C00.total, "3-Sigma Feature Compliment", vlim)
    #map_vis.save(fig, directory + 'OneOff/three_sigma_mask_compliment_' + map_type + '.pdf')
    state_object_Call_m_C00.total[three_sigma_mask] = three_sigma_values

    three_sigma_compliment_values = state_object_Call_m_C00.total[three_sigma_mask_compliment]
    state_object_Call_m_C00.total[three_sigma_mask_compliment] = 0.0
    fig, ax = map_vis.plot_stateObject(state_object_Call_m_C00.total, "3-Sigma Features", vlim)
    #map_vis.save(fig, directory + 'OneOff/three_sigma_mask_' + map_type + '.pdf')
    state_object_Call_m_C00.total[three_sigma_mask_compliment] = three_sigma_compliment_values


map_vis = VisualizationBase(halt_formatting=True)

directory = os.path.abspath('.') +"/Plots/OneOff/"
os.makedirs(directory, exist_ok=True)

def main():
    map_type = 'sphere'
    map_type = 'surface'
    acceleration_distribution_plots(map_type)
    acceleration_histogram_std(map_type)
    cumulative_distribution(map_type)
    # acceleration_masks('sphere')
   
    plt.show()



if __name__ == "__main__":
    main()