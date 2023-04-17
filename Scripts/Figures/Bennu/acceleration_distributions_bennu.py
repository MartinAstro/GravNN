import os

import matplotlib.pyplot as plt
import numpy as np

from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Support.StateObject import StateObject
from GravNN.Trajectories import DHGridDist, SurfaceDist
from GravNN.Visualization.MapBase import MapBase
from GravNN.Visualization.VisualizationBase import VisualizationBase


def std_masks(stateObject, sigma):
    sigma_mask = np.where(
        stateObject.total
        > (np.mean(stateObject.total) + sigma * np.std(stateObject.total)),
    )
    sigma_mask_compliment = np.where(
        stateObject.total
        < (np.mean(stateObject.total) + sigma * np.std(stateObject.total)),
    )
    return sigma_mask, sigma_mask_compliment


def get_poly_state_object(map_type="sphere"):
    planet = Bennu()
    obj_file = planet.obj_file
    density_deg = 180

    radius_min = planet.radius
    if map_type == "sphere":
        trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    else:
        trajectory = SurfaceDist(planet, obj_file)
    Clm_r0_gm = Polyhedral(planet, obj_file, trajectory)
    Clm_a = Clm_r0_gm.load().accelerations

    return StateObject(trajectory, Clm_a)


def get_sh_state_object_C00(map_type="sphere"):
    planet = Bennu()
    obj_file = planet.obj_file
    density_deg = 180

    radius_min = planet.radius
    if map_type == "sphere":
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
    # map_vis.save(plt.gcf(), directory + "a_norm_hist.pdf")

    state_object_Call_m_C00 = state_object_true - state_object_C00
    plt.figure()
    plt.hist(state_object_Call_m_C00.total.reshape((-1,)), 100)
    plt.xlabel("||a-a0|| [m/s^2]")
    plt.ylabel("Frequency")
    # map_vis.save(plt.gcf(), directory + "a_m_C00_norm_hist.pdf")


def cumulative_distribution(map_type):
    state_object_true = get_poly_state_object(map_type)
    state_object_C00 = get_sh_state_object_C00(map_type)

    state_object_Call_m_C00 = state_object_true - state_object_C00
    sorted_data = np.sort(state_object_Call_m_C00.total.reshape((-1,)))

    len_data = len(sorted_data)
    data = state_object_Call_m_C00.total
    data_mean = np.mean(data)
    data_std = np.std(data)
    total = len_data - len(np.where(data > (data_mean + 1 * data_std))[0])
    total2 = len_data - len(np.where(data > (data_mean + 2 * data_std))[0])
    total3 = len_data - len(np.where(data > (data_mean + 3 * data_std))[0])

    x = np.arange(0, len(sorted_data), 1)
    y = np.zeros(x.shape)
    y0 = 0
    for i in range(len(x)):
        y0 += sorted_data[i]
        y[i] = y0

    plt.figure()
    plt.plot(x, y)
    plt.ylabel("Cumulative Sum")
    plt.xlabel("Data Number")

    plt.figure()
    plt.plot(x, np.gradient(y))
    plt.vlines(total, 0, np.max(np.gradient(y)), colors="r", label="1STD")
    plt.vlines(total2, 0, np.max(np.gradient(y)), colors="g", label="2STD")
    plt.vlines(total3, 0, np.max(np.gradient(y)), colors="y", label="3STD")
    plt.ylabel("Cumulative Sum Gradient")
    plt.xlabel("Data Number")


def acceleration_histogram_std(map_type):
    # Distribution Stats

    state_object_true = get_poly_state_object(map_type)
    state_object_C00 = get_sh_state_object_C00(map_type)

    state_object_Call_m_C00 = state_object_true - state_object_C00
    plt.figure()
    plt.hist(state_object_Call_m_C00.total.reshape((-1,)), 1000)
    plt.vlines(
        np.mean(state_object_Call_m_C00.total),
        0,
        5000,
        colors="r",
        label="mean",
    )
    plt.vlines(
        np.mean(state_object_Call_m_C00.total) + np.std(state_object_Call_m_C00.total),
        0,
        5000,
        colors="g",
        label="mean+std",
    )
    plt.vlines(
        np.mean(state_object_Call_m_C00.total)
        + 2 * np.std(state_object_Call_m_C00.total),
        0,
        5000,
        colors="y",
        label="mean+2std",
    )
    plt.vlines(1e-8, 0, 5000, colors="y", label="SRP at Earth")
    plt.xlabel("||a-a0|| [m/s^2]")


def acceleration_masks(map_type):
    state_object_true = get_poly_state_object(map_type)
    state_object_C00 = get_sh_state_object_C00(map_type)

    state_object_Call_m_C00 = state_object_true - state_object_C00
    sigma_5_mask, sigma_5_mask_compliment = std_masks(state_object_Call_m_C00, 5)
    sigma_4_mask, sigma_4_mask_compliment = std_masks(state_object_Call_m_C00, 4)
    sigma_3_mask, sigma_3_mask_compliment = std_masks(
        state_object_Call_m_C00,
        3,
    )
    sigma_2_mask, sigma_2_mask_compliment = std_masks(state_object_Call_m_C00, 2)
    sigma_1_mask, sigma_1_mask_compliment = std_masks(state_object_Call_m_C00, 1)

    # plt.figure()
    # data = state_object_Call_m_C00.total
    # plt.hist(data[sigma_5_mask].reshape((-1,)), 100)
    # plt.hist(data[sigma_5_mask_compliment].reshape((-1,)), 100)

    map_vis = MapBase(unit="mGal")
    plt.rc("text", usetex=False)

    layer_stateObject = state_object_true - state_object_C00
    layer_stateObject.total[sigma_1_mask_compliment] = 0.0
    layer_stateObject.total[sigma_1_mask] = 1.0
    layer_stateObject.total[sigma_2_mask] = 2.0
    layer_stateObject.total[sigma_3_mask] = 3.0
    layer_stateObject.total[sigma_4_mask] = 4.0
    layer_stateObject.total[sigma_5_mask] = 5.0
    map_vis.plot_stateObject(layer_stateObject.total, "sigma map")

    vlim = [0, np.max(state_object_Call_m_C00.total) * 10000]

    os.path.abspath(".") + "/Plots/"
    fig, ax = map_vis.plot_stateObject(state_object_Call_m_C00.total, "Truth", vlim)
    # map_vis.save(fig, directory + 'OneOff/true_' + map_type + '.pdf')

    sigma_2_values = state_object_Call_m_C00.total[sigma_2_mask]
    state_object_Call_m_C00.total[sigma_2_mask] = 0.0
    fig, ax = map_vis.plot_stateObject(
        state_object_Call_m_C00.total,
        "2-Sigma Feature Compliment",
        vlim,
    )
    # map_vis.save(fig, directory + f'OneOff/sigma_2_mask_compliment_{map_type}.pdf')
    state_object_Call_m_C00.total[sigma_2_mask] = sigma_2_values

    sigma_2_compliment_values = state_object_Call_m_C00.total[sigma_2_mask_compliment]
    state_object_Call_m_C00.total[sigma_2_mask_compliment] = 0.0
    fig, ax = map_vis.plot_stateObject(
        state_object_Call_m_C00.total,
        "2-Sigma Features",
        vlim,
    )
    # map_vis.save(fig, directory + f'OneOff/sigma_2_mask_{map_type}.pdf')
    state_object_Call_m_C00.total[sigma_2_mask_compliment] = sigma_2_compliment_values

    sigma_3_values = state_object_Call_m_C00.total[sigma_3_mask]
    state_object_Call_m_C00.total[sigma_3_mask] = 0.0
    fig, ax = map_vis.plot_stateObject(
        state_object_Call_m_C00.total,
        "3-Sigma Feature Compliment",
        vlim,
    )
    # map_vis.save(fig, directory + f'OneOff/sigma_3_mask_compliment_{map_type}.pdf')
    state_object_Call_m_C00.total[sigma_3_mask] = sigma_3_values

    sigma_3_compliment_values = state_object_Call_m_C00.total[sigma_3_mask_compliment]
    state_object_Call_m_C00.total[sigma_3_mask_compliment] = 0.0
    fig, ax = map_vis.plot_stateObject(
        state_object_Call_m_C00.total,
        "3-Sigma Features",
        vlim,
    )
    # map_vis.save(fig, directory + f'OneOff/sigma_3_mask_{map_type}.pdf')
    state_object_Call_m_C00.total[sigma_3_mask_compliment] = sigma_3_compliment_values


map_vis = VisualizationBase(halt_formatting=True)

directory = os.path.abspath(".") + "/Plots/OneOff/"
os.makedirs(directory, exist_ok=True)


def main():
    map_type = "sphere"
    map_type = "surface"
    acceleration_distribution_plots(map_type)
    acceleration_histogram_std(map_type)
    cumulative_distribution(map_type)
    # acceleration_masks('sphere')

    plt.show()


if __name__ == "__main__":
    main()
