import os
import numpy as np

from GravNN.Support.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.CelestialBodies.Planets import Earth, Moon
from GravNN.Trajectories import DHGridDist

import matplotlib.pyplot as plt

def calculate_entropy(data, max=100):
    from scipy.stats import entropy
    n, bins, patches = plt.hist(data, 1000)
    scipy_entropy = np.round(entropy(n/np.sum(n)),2)

    values, edges = np.histogram(data, bins=1000, range=[0, max])
    numpy_entropy = np.round(entropy(values/np.sum(values)),2)

    print("Scipy Entropy:" + str(scipy_entropy))
    print("Numpy Entropy:" + str(numpy_entropy))
    return numpy_entropy

def std_masks(grid, sigma):
    sigma_mask = np.where(grid.total > (np.mean(grid.total) + sigma*np.std(grid.total)))
    sigma_mask_compliment = np.where(grid.total < (np.mean(grid.total) + sigma*np.std(grid.total)))
    return sigma_mask, sigma_mask_compliment

def get_grid(degree):
    planet = Moon()
    model_file = planet.sh_file
    density_deg = 180

    radius_min = planet.radius
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    Clm_r0_gm = SphericalHarmonics(model_file, degree=degree, trajectory=trajectory)
    Clm_a = Clm_r0_gm.load().accelerations
    return Grid(trajectory, Clm_a)

def acceleration_histogram_masks():

    grid_true = get_grid(1000)
    grid_C00 = get_grid(0)
    grid_C22 = get_grid(2)

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

    grid_true = get_grid(1000)
    grid_C00 = get_grid(0)
    grid_C22 = get_grid(2)
    grid_C33 = get_grid(3)
    grid_C4040 = get_grid(40)

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
    grid_true = get_grid(1000)
    grid_C00 = get_grid(0)
    grid_C22 = get_grid(2)

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
    
    grid_true = get_grid(1000)
    grid_C00 = get_grid(0)
    grid_C22 = get_grid(2)

    grid_Call_m_C22 = grid_true - grid_C22
    map_vis = MapVisualization(unit='mGal')
    map_vis.fig_size = map_vis.half_page

    map_vis.newFig()
    data = grid_Call_m_C22.total.reshape((-1,))*10000
    plt.xlim([0, (1E-2)*10000])
    plt.xlabel('$|\delta \mathbf{a}|$' + '[mGal]')
    plt.ylabel('Frequency')
    entropy = calculate_entropy(data)
    plt.legend(['Entropy = ' + str(entropy)])
    map_vis.save(plt.gcf(), directory + 'moon_acc_histogram.pdf')


def acceleration_histogram_scaled():
    # Distribution Stats 
    grid_true = get_grid(1000)
    grid_C22 = get_grid(2)

    grid_Call_m_C22 = grid_true - grid_C22

    a = grid_Call_m_C22.acceleration
    from GravNN.Preprocessors import UniformScaler
    a_transformer = UniformScaler()
    a_transformed = a_transformer.fit_transform(a)

    planet = Moon()
    density_deg = 180
    radius_min = planet.radius
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)

    grid_transformed = Grid(trajectory, a_transformed)
    map_vis = MapVisualization(unit='m/s^2')
    map_vis.fig_size = map_vis.half_page

    map_vis.newFig()
    data = grid_transformed.total.reshape((-1,))
    plt.xlabel('Transformed $|\delta \mathbf{a\'}|$')
    plt.ylabel('Frequency')
    plt.xlim([0,1])

    entropy = calculate_entropy(data, max=1)
    plt.legend(['Entropy = ' + str(entropy)])
    map_vis.save(plt.gcf(), directory + 'moon_acc_transformed_histogram.pdf')

def acceleration_masks():
    grid_true = get_grid(1000)
    grid_C00 = get_grid(0)
    grid_C22 = get_grid(2)

    grid_Call_m_C22 = grid_true - grid_C22
    five_sigma_mask, five_sigma_mask_compliment = std_masks(grid_Call_m_C22, 5)
    four_sigma_mask, four_sigma_mask_compliment = std_masks(grid_Call_m_C22, 4)
    three_sigma_mask, three_sigma_mask_compliment = std_masks(grid_Call_m_C22, 3)
    two_sigma_mask, two_sigma_mask_compliment = std_masks(grid_Call_m_C22, 2)
    one_sigma_mask, one_sigma_mask_compliment = std_masks(grid_Call_m_C22, 1)

    # plt.figure()
    # plt.hist(grid_Call_m_C22.total[five_sigma_mask].reshape((-1,)), 100)
    # plt.hist(grid_Call_m_C22.total[five_sigma_mask_compliment].reshape((-1,)), 100)

    map_vis = MapVisualization(unit='mGal')
    plt.rc('text', usetex=True)
    map_vis.fig_size = map_vis.half_page
    map_vis.tick_interval = [60,60]

    layer_grid = grid_true - grid_C22
    layer_grid.total[one_sigma_mask_compliment] = 0.0
    layer_grid.total[one_sigma_mask] = 1.0
    layer_grid.total[two_sigma_mask] = 2.0
    layer_grid.total[three_sigma_mask] = 3.0
    layer_grid.total[four_sigma_mask] = 4.0
    layer_grid.total[five_sigma_mask] = 5.0
    fig, ax = map_vis.plot_grid(layer_grid.total, 'sigma map')#, cmap='Dark2')
    map_vis.save(fig, directory + 'moon_sigma_map.pdf')
    # vlim = [0, np.max(grid_Call_m_C22.total)*10000]
    
    # directory = os.path.abspath('.') +"/Plots/"
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "Truth", vlim)
    # #map_vis.save(fig, directory + 'OneOff/true_' +  + '.pdf')

    # one_sigma_values = grid_Call_m_C22.total[one_sigma_mask]
    # grid_Call_m_C22.total[one_sigma_mask] = 0.0
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "1-Sigma Feature Compliment", vlim)
    # #map_vis.save(fig, directory + 'OneOff/one_sigma_mask_compliment_' +  + '.pdf')
    # grid_Call_m_C22.total[one_sigma_mask] = one_sigma_values

    # one_sigma_compliment_values = grid_Call_m_C22.total[one_sigma_mask_compliment]
    # grid_Call_m_C22.total[one_sigma_mask_compliment] = 0.0
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "1-Sigma Features", vlim)
    # #map_vis.save(fig, directory + 'OneOff/one_sigma_mask_' +  + '.pdf')
    # grid_Call_m_C22.total[one_sigma_mask_compliment] = one_sigma_compliment_values

    # two_sigma_values = grid_Call_m_C22.total[two_sigma_mask]
    # grid_Call_m_C22.total[two_sigma_mask] = 0.0
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "2-Sigma Feature Compliment", vlim)
    # #map_vis.save(fig, directory + 'OneOff/two_sigma_mask_compliment_' +  + '.pdf')
    # grid_Call_m_C22.total[two_sigma_mask] = two_sigma_values

    # two_sigma_compliment_values = grid_Call_m_C22.total[two_sigma_mask_compliment]
    # grid_Call_m_C22.total[two_sigma_mask_compliment] = 0.0
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "2-Sigma Features", vlim)
    # #map_vis.save(fig, directory + 'OneOff/two_sigma_mask_' +  + '.pdf')
    # grid_Call_m_C22.total[two_sigma_mask_compliment] = two_sigma_compliment_values

    # three_sigma_values = grid_Call_m_C22.total[three_sigma_mask]
    # grid_Call_m_C22.total[three_sigma_mask] = 0.0
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "3-Sigma Feature Compliment", vlim)
    # #map_vis.save(fig, directory + 'OneOff/three_sigma_mask_compliment_' +  + '.pdf')
    # grid_Call_m_C22.total[three_sigma_mask] = three_sigma_values

    # three_sigma_compliment_values = grid_Call_m_C22.total[three_sigma_mask_compliment]
    # grid_Call_m_C22.total[three_sigma_mask_compliment] = 0.0
    # fig, ax = map_vis.plot_grid(grid_Call_m_C22.total, "3-Sigma Features", vlim)
    # #map_vis.save(fig, directory + 'OneOff/three_sigma_mask_' +  + '.pdf')
    # grid_Call_m_C22.total[three_sigma_mask_compliment] = three_sigma_compliment_values


map_vis = VisualizationBase()
map_vis.fig_size = map_vis.half_page

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12.0)

#map_vis.fig_size = (3, 1.8)

directory = os.path.abspath('.') +"/Plots/OneOff/"
os.makedirs(directory, exist_ok=True)

def main():

    #acceleration_distribution_plots()
    #acceleration_histogram_masks()
    acceleration_histogram_std()
    acceleration_histogram_scaled()
    #cumulative_distribution()
    #acceleration_masks()
   
    plt.show()



if __name__ == "__main__":
    main()