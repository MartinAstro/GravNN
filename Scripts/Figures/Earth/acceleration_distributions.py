import os

import matplotlib.pyplot as plt
import numpy as np

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Preprocessors import UniformScaler
from GravNN.Support.Grid import Grid
from GravNN.Trajectories import DHGridDist
from GravNN.Visualization.MapBase import MapBase
from GravNN.Visualization.VisualizationBase import VisualizationBase


def calculate_entropy(data, max=100):
    from scipy.stats import entropy

    n, bins, patches = plt.hist(data, 1000)
    scipy_entropy = np.round(entropy(n / np.sum(n)), 2)

    values, edges = np.histogram(data, bins=1000, range=[0, max])
    numpy_entropy = np.round(entropy(values / np.sum(values)), 2)

    print("Scipy Entropy:" + str(scipy_entropy))
    print("Numpy Entropy:" + str(numpy_entropy))
    return numpy_entropy


def std_masks(grid, sigma):
    max_value = np.mean(grid.total) + sigma * np.std(grid.total)
    sigma_mask = np.where(grid.total > max_value)
    sigma_mask_compliment = np.where(grid.total < max_value)
    return sigma_mask, sigma_mask_compliment


def get_grid(degree):
    planet = Earth()
    model_file = planet.sh_file
    density_deg = 180

    radius_min = planet.radius
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    Clm_r0_gm = SphericalHarmonics(model_file, degree=degree, trajectory=trajectory)
    Clm_a = Clm_r0_gm.load().accelerations
    return Grid(trajectory, Clm_a)


def acceleration_magnitude_histogram():
    grid_true = get_grid(1000)
    grid_C00 = get_grid(0)
    grid_C22 = get_grid(2)

    grid_da_00 = grid_true - grid_C00
    grid_da_22 = grid_true - grid_C22

    map_vis.newFig()
    plt.hist(grid_true.total.reshape((-1,)), 100)
    plt.xlabel(r"$||a||$ [m/$s^2$]")
    plt.ylabel("Frequency")
    map_vis.save(plt.gcf(), directory + "a_norm_hist.pdf")

    map_vis.newFig()
    plt.hist(grid_da_00.total.reshape((-1,)), 100)
    plt.xlabel(r"$||a-a_0||$ [m/$s^2$]")
    plt.ylabel("Frequency")

    map_vis.newFig()
    plt.hist(grid_da_22.total.reshape((-1,)), 100)
    plt.xlabel(r"$||a-(a_0 + \nabla(U_{c_{22}}))||$ [m/$s^2$]")
    plt.ylabel("Frequency")
    map_vis.save(plt.gcf(), directory + "a_norm_m_grad_C22_hist.pdf")


def acceleration_cumsum():
    """Sort the acceleration data. Add it incrementally
    and identify the indices at which the transition to 1,2,3 sigma
    occurs. Also plot the gradients
    """
    grid_true = get_grid(1000)
    grid_C22 = get_grid(2)

    grid_da = grid_true - grid_C22
    da_mean = np.mean(grid_da.total)
    sigma = np.std(grid_da.total)

    # Sort accelerations
    sorted_data = np.sort(grid_da.total.reshape((-1,)))

    # Identify the index at which acc > sigma
    N = len(sorted_data)
    sigma_1_idx = N - len(np.where(grid_da.total > (da_mean + 1 * sigma))[0])
    sigma_2_idx = N - len(np.where(grid_da.total > (da_mean + 2 * sigma))[0])
    sigma_3_idx = N - len(np.where(grid_da.total > (da_mean + 3 * sigma))[0])

    # Generate data for cumsum plot
    x = np.arange(0, N, 1)
    y = np.zeros(x.shape)
    y0 = 0
    for i in range(len(x)):
        y0 += sorted_data[i]
        y[i] = y0

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, y)
    plt.ylabel("Cumulative Sum")
    plt.xlabel("Data Number")

    plt.subplot(2, 1, 2)
    plt.plot(x, np.gradient(y))
    plt.vlines(sigma_1_idx, 0, np.max(np.gradient(y)), colors="r", label="1STD")
    plt.vlines(sigma_2_idx, 0, np.max(np.gradient(y)), colors="g", label="2STD")
    plt.vlines(sigma_3_idx, 0, np.max(np.gradient(y)), colors="y", label="3STD")
    plt.ylabel("Cumulative Sum Gradient")
    plt.xlabel("Data Number")


def acceleration_histogram_std():
    grid_true = get_grid(1000)
    grid_C22 = get_grid(2)

    grid_da = grid_true - grid_C22

    map_vis = MapBase(unit="m/s^2")
    map_vis.fig_size = map_vis.half_page_default

    map_vis.newFig()
    data = grid_da.total.reshape((-1,))
    entropy = calculate_entropy(data)
    plt.xlim([0, (1e-2)])
    plt.xlabel("$|\delta \mathbf{a}|$" + "[m/$s^2$]")
    plt.ylabel("Frequency")
    plt.legend(["Entropy = " + str(entropy)])

    map_vis.save(plt.gcf(), directory + "earth_acc_histogram.pdf")


def acceleration_histogram_scaled():
    grid_true = get_grid(1000)
    grid_C22 = get_grid(2)

    grid_da = grid_true - grid_C22

    a = grid_da.acceleration

    a_transformer = UniformScaler()
    a_transformed = a_transformer.fit_transform(a)

    planet = Earth()
    density_deg = 180
    radius_min = planet.radius
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    grid_transformed = Grid(trajectory, a_transformed)

    map_vis = MapBase(unit="m/s^2")
    map_vis.fig_size = map_vis.half_page_default

    map_vis.newFig()
    data = grid_transformed.total.reshape((-1,))
    entropy = calculate_entropy(data, max=1)
    plt.xlim([0, 1])
    plt.xlabel("Transformed $|\delta \mathbf{a'}|$")
    plt.ylabel("Frequency")
    plt.legend(["Entropy = " + str(entropy)])

    map_vis.save(plt.gcf(), directory + "earth_acc_transformed_histogram.pdf")


def map_as_sigmas():
    grid_true = get_grid(1000)
    grid_C22 = get_grid(2)

    grid_da = grid_true - grid_C22
    sigma_5_mask, sigma_5_mask_compliment = std_masks(grid_da, 5)
    sigma_4_mask, sigma_4_mask_compliment = std_masks(grid_da, 4)
    sigma_3_mask, sigma_3_mask_compliment = std_masks(grid_da, 3)
    sigma_2_mask, sigma_2_mask_compliment = std_masks(grid_da, 2)
    sigma_1_mask, sigma_1_mask_compliment = std_masks(grid_da, 1)

    map_vis = MapBase(unit="m/s^2")
    map_vis.fig_size = map_vis.half_page_default
    map_vis.tick_interval = [60, 60]

    plt.rc("text", usetex=True)

    layer_grid = grid_true - grid_C22
    layer_grid.total[sigma_1_mask_compliment] = 0
    layer_grid.total[sigma_1_mask] = 1
    layer_grid.total[sigma_2_mask] = 2
    layer_grid.total[sigma_3_mask] = 3
    layer_grid.total[sigma_4_mask] = 4
    layer_grid.total[sigma_5_mask] = 5
    fig, ax = map_vis.plot_grid(layer_grid.total, "sigma map")  # , cmap='Dark2')
    map_vis.save(fig, directory + "earth_sigma_map.pdf")


map_vis = VisualizationBase()
map_vis.fig_size = map_vis.half_page_default

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=12.0)

directory = os.path.abspath(".") + "/Plots/OneOff/"
os.makedirs(directory, exist_ok=True)


def main():
    acceleration_magnitude_histogram()
    acceleration_histogram_std()
    acceleration_histogram_scaled()
    acceleration_cumsum()
    map_as_sigmas()

    plt.show()


if __name__ == "__main__":
    main()
