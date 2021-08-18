        
import os
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Trajectories import DHGridDist, RandomDist
from GravNN.Support.Grid import Grid
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.DataVisualization import DataVisualization
from GravNN.Support.transformations import cart2sph, sphere2cart, project_acceleration

def get_title(config):
    title = str(config['PINN_constraint_fcn'][0]) + "-" + \
        str(config['N_train'][0]) + "-" + \
        str(config['radius_max'][0] - config['planet'][0].radius)
    return title

def format_potential_as_Nx3(u):
    U_Nx3 = np.zeros((len(u), 3))
    U_Nx3[:,0] = u
    return U_Nx3

def minmax(values):
    print(np.min(values, axis=0))
    print(np.max(values, axis=0))

def generate_nn_data(x, model, config):
    x_transformer = config['x_transformer'][0]
    a_transformer = config['a_transformer'][0]
    u_transformer = config['u_transformer'][0]
    if config['basis'][0] == 'spherical':
        x = cart2sph(x)
        x[:,1:3] = np.deg2rad(x[:,1:3])

    x = x_transformer.transform(x)
    u_pred, a_pred, laplace_pred, curl_pred = model.output((x,x))
    u_pred = u_transformer.inverse_transform(u_pred)
    a_pred = a_transformer.inverse_transform(a_pred)
    return u_pred, a_pred

def plot_acceleration_comp(map_vis, grid_acc, title):
    map_vis.newFig()
    plt.subplot(3, 1, 1)
    map_vis.plot_grid(grid_acc.r, "Ar", new_fig=False)   
    plt.subplot(3, 1, 2)
    map_vis.plot_grid(grid_acc.theta, "Atheta", new_fig=False, vlim=[-0.02, 0.02])   
    plt.subplot(3, 1, 3)
    map_vis.plot_grid(grid_acc.phi, "Aphi", new_fig=False,  vlim=[-0.0025, 0.0025])   
    plt.suptitle(title)

def plot_potential_full_C22(map_vis, grid_pot_true, grid_C22):
    map_vis.newFig()
    plt.subplot(2,1,1)
    map_vis.plot_grid(grid_pot_true.total, "Potential True", new_fig=False)
    plt.subplot(2,1,2)
    map_vis.plot_grid((grid_pot_true - grid_C22).total, "Potential C22 Removed",new_fig=False)


def plot_radial_potential_dist(map_vis, trajectory, model, config, title):
    x, a, u = get_sh_data(trajectory, config['grav_file'][0], max_deg=1000, deg_removed=-1)
    u3, a_pred = generate_nn_data(x,a,model, config)    

    x_sph = cart2sph(x)
    a_sph = project_acceleration(x_sph, np.array(a, dtype=float))
    a_sph_pred = project_acceleration(x_sph, np.array(a_pred, dtype=float))

    map_vis.newFig()
    plt.subplot(2,2,1)
    plt.scatter(x_sph[:,0], u3[:,0], s=1, c='b', label='Pred')
    plt.scatter(x_sph[:,0], u, s=1, c='r', label='True')
    plt.ylabel("potential")
    plt.legend()

    plt.subplot(2,2,2)
    plt.scatter(x_sph[:,0], u3[:,0] - u[:,0], s=1)
    plt.ylabel("residual")

    plt.subplot(2,2,3)
    plt.scatter(x_sph[:,0], a_sph_pred[:,0], s=1, c='b', label='Pred')
    plt.scatter(x_sph[:,0], a_sph[:,0], s=1, c='r', label='True')
    plt.ylabel("acceleration")

    plt.subplot(2,2,4)
    plt.scatter(x_sph[:,0], a_sph[:,0] - a_sph_pred[:,0], s=1)
    plt.ylabel("residual")
    plt.suptitle(title)

def get_spherical_data(x, a):
    x_sph = cart2sph(x)
    a_sph = project_acceleration(x_sph, np.array(a, dtype=float))
    return x_sph, a_sph

def get_vlim_bounds(dist, sigma):
    mu = np.mean(dist)
    std = np.std(dist)
    vlim_min = clamp(mu-sigma*std, 0, np.inf)
    vlim_max = mu+sigma*std
    return [vlim_min, vlim_max]

def clamp(n, smallest, largest): 
    return max(smallest, min(n, largest))

def true_pred_acc_pot_plots(map_vis, grid_pot_true, grid_pot_pred, grid_acc_true, grid_acc_pred, title):
    map_vis.newFig()
    plt.subplot(2,3,1)
    map_vis.plot_grid(grid_pot_true.total, "Potential True", new_fig=False, ticks=False)
    ax = plt.gcf().axes[0]
    plt.subplot(2,3,2)
    map_vis.plot_grid(grid_pot_pred.total, "Potential NN", new_fig=False, ticks=False)#, vlim=[ax.images[0].colorbar.vmin, ax.images[0].colorbar.vmax])
    plt.subplot(2,3,3)
    map_vis.plot_grid((grid_pot_true - grid_pot_pred).total, "Pot Diff", new_fig=False, ticks=False)#, vlim=[0,10000])

    plt.subplot(2,3,4)
    map_vis.plot_grid(grid_acc_true.total, "Acceleration True", new_fig=False, ticks=False)   
    ax = plt.gcf().axes[6]
    plt.subplot(2,3,5)
    map_vis.plot_grid(grid_acc_pred.total, "Acceleration NN", new_fig=False, ticks=False, vlim=[ax.images[0].colorbar.vmin, ax.images[0].colorbar.vmax])
    plt.subplot(2,3,6)
    map_vis.plot_grid((grid_acc_true - grid_acc_pred).total, "Acc Diff", new_fig=False, ticks=False, vlim=[0, 0.025])
    plt.suptitle(title)

def main():
    
    directory = os.path.abspath('.') +"/Plots/OneOff/"
    os.makedirs(directory, exist_ok=True)

    mapUnit = 'm/s^2'
    map_vis = MapVisualization(mapUnit, halt_formatting=True)
    map_vis.fig_size = map_vis.half_page
    data_vis = DataVisualization(mapUnit, halt_formatting=True)

    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000

    radius_min = planet.radius
    DH_trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    rand_trajectory = RandomDist(planet, [radius_min, planet.radius + 420000], points=20000)

    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=DH_trajectory).load(override=False)
    C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=DH_trajectory).load(override=False)

    u_3vec = np.zeros(np.shape(Call_r0_gm.accelerations))
    u_3vec[:,0] = Call_r0_gm.potentials
    grid_pot_true = Grid(trajectory=DH_trajectory, accelerations=u_3vec, transform=False)
    grid_acc_true = Grid(trajectory=DH_trajectory, accelerations=Call_r0_gm.accelerations)

    u_3vec[:,0] = C22_r0_gm.potentials
    grid_C22 = Grid(trajectory=DH_trajectory, accelerations=u_3vec)

    #plot_acceleration_comp(map_vis, grid_acc_true)
    #plot_potential_full_C22(map_vis, grid_pot_true, grid_C22)

    df = pd.read_pickle('Data/Dataframes/useless_070121_v1.data')
    model_ids = df['id'].values[:]
    for model_id in model_ids:
        config, model = load_config_and_model(model_id, df)
        title = get_title(config)
        
        x = Call_r0_gm.positions
        a = Call_r0_gm.accelerations
        u = Call_r0_gm.potentials
  
        u_pred, a_pred = generate_nn_data(x, model, config)
        x_sph, a_sph = get_spherical_data(x, a)
        x_sph, a_sph_pred = get_spherical_data(x, a_pred)

        data_vis.plot_potential_dist(x_sph, u, u_pred, "Potential Error (w.r.t. r)")
        data_vis.plot_acceleration_dist(x, a, a_pred, "Cartesian Acceleration Error")
        data_vis.plot_acceleration_dist(x_sph, a_sph, a_sph_pred, "Spherical Acceleration Error")

        grid_pot_pred = Grid(trajectory=DH_trajectory, accelerations=format_potential_as_Nx3(u), transform=False)
        grid_acc_pred = Grid(trajectory=DH_trajectory, accelerations=a_pred)

        true_pred_acc_pot_plots(map_vis, grid_pot_true, grid_pot_pred, grid_acc_true, grid_acc_pred, title)
        plot_acceleration_comp(map_vis, grid_acc_pred, title)

    plt.show()

if __name__ == "__main__":
    main()
