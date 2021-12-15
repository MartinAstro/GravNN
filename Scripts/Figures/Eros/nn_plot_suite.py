        
from GravNN.Trajectories.SurfaceDist import SurfaceDist
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Eros, Toutatis
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.Trajectories import DHGridDist, RandomDist, RandomAsteroidDist
from GravNN.Support.Grid import Grid
from GravNN.Networks.Model import load_config_and_model
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.PolyVisualization import PolyVisualization
from GravNN.Visualization.DataVisualization import DataVisualization
from GravNN.Networks.Data import single_training_validation_split
from GravNN.Support.transformations import cart2sph, sphere2cart, project_acceleration
from GravNN.Visualization.FigureSupport import get_vlim_bounds, format_potential_as_Nx3
from GravNN.Visualization.MapVisSuite import MapVisSuite
from GravNN.Visualization.DataVisSuite import DataVisSuite

def make_fcn_name_latex_compatable(name):
    components = name.split("_")
    return components[0] + r"$_{" + components[1] + r"}$"

def get_title(config):
    fcn_name = make_fcn_name_latex_compatable(config['PINN_constraint_fcn'][0].__name__)
    title = fcn_name + " " + \
        str(config['N_train'][0]) + " " + \
        str(config['radius_max'][0] - config['planet'][0].radius)
    return title

def minmax(values):
    print(np.min(values, axis=0))
    print(np.max(values, axis=0))

def get_spherical_data(x, a):
    x_sph = cart2sph(x)
    a_sph = project_acceleration(x_sph, np.array(a, dtype=float))
    return x_sph, a_sph

def main():
    directory = os.path.abspath('.') +"/Plots/Asteroid/"
    os.makedirs(directory, exist_ok=True)

    map_vis = MapVisualization(halt_formatting=False)
    map_vis.fig_size = map_vis.half_page

    poly_vis = PolyVisualization(halt_formatting=False)

    data_vis = DataVisualization(halt_formatting=False)
    data_vis.fig_size = data_vis.full_page

    data_vis = DataVisSuite(halt_formatting=False)
    data_vis.fig_size = data_vis.full_page

    map_suite = MapVisSuite(halt_formatting=False)
    map_suite.fig_size = map_suite.full_page

    planet = Eros()
    # planet = Toutatis()

    density_deg = 180

    radius_min = planet.radius
    DH_trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    train_trajectory = RandomAsteroidDist(planet, [0.0, planet.radius + 5000], points=50000, grav_file=[planet.obj_8k])
    surface_trajectory = SurfaceDist(planet, planet.obj_8k)

    poly_gm = Polyhedral(planet, planet.obj_8k, trajectory=DH_trajectory).load(override=False)
    surface_poly_gm = Polyhedral(planet, planet.obj_8k, trajectory=surface_trajectory).load(override=False)

    u_3vec = np.zeros(np.shape(poly_gm.accelerations))
    u_3vec[:,0] = poly_gm.potentials
    grid_pot_true = Grid(trajectory=DH_trajectory, accelerations=u_3vec, transform=False)
    grid_acc_true = Grid(trajectory=DH_trajectory, accelerations=poly_gm.accelerations)

    #map_suite.plot_acceleration_comp(grid_acc_true, None)


    #plt.show()
    # df = pd.read_pickle('Data/Dataframes/useless_070221_v4.data')
    df = pd.read_pickle('Data/Dataframes/useless_072321_v1.data')
    model_ids = df['id'].values[:]
    for model_id in model_ids:
        config, model = load_config_and_model(model_id, df)
        title = get_title(config)

        N_extra = config.get('extra_N_train', 0)
        directory = os.path.abspath('.') +"/Plots/Asteroid/" + config['PINN_constraint_fcn'][0].__name__ + "/" + str(np.round(config['radius_min'][0],2)) + "_" +str(np.round(config['radius_max'][0],2)) + "_"+str(N_extra) + "/"
        os.makedirs(directory, exist_ok=True)

        train_trajectory = config['distribution'][0](config['planet'][0], [config['radius_min'][0], config['radius_max'][0]], config['N_dist'][0], **config)
        training_poly_gm = Polyhedral(planet, planet.obj_8k, trajectory=train_trajectory).load(override=False)

        # x_train, x_val = single_training_validation_split(train_trajectory.positions, N_train=2500, N_val=0) 
        # poly_vis.plot_polyhedron(poly_gm.mesh, np.linalg.norm(surface_poly_gm.accelerations, axis=1))
        # poly_vis.save(plt.gcf(), directory+"Asteroid_Surface.pdf")

        # poly_vis.plot_position_data(x_train)
        # poly_vis.save(plt.gcf(), directory+"Asteroid_Training.pdf")

        test_trajectory = config['distribution'][0](config['planet'][0], [0, planet.radius + 10000], config['N_dist'][0], **config)        
        test_poly_gm = Polyhedral(planet, planet.obj_8k, trajectory=test_trajectory).load(override=False)

        x = test_poly_gm.positions
        a = test_poly_gm.accelerations
        u = test_poly_gm.potentials
  
        data_pred = model.generate_nn_data(x)
        a_pred = data_pred['a']
        u_pred = data_pred['u']

        x_sph, a_sph = get_spherical_data(x, a)
        x_sph, a_sph_pred = get_spherical_data(x, a_pred)

        data_vis.plot_potential_dist(x_sph, u, u_pred, "Potential Error (w.r.t. r)")
        # data_vis.plot_acceleration_dist(x, a, a_pred, "Cartesian Acceleration Error")
        data_vis.plot_acceleration_dist(x_sph, a_sph, a_sph_pred, "Spherical Acceleration Error", vlines=[planet.radius, config['radius_min'][0], config['radius_max'][0]])
        data_vis.save(plt.gcf(), directory+"Rand_Acc_Components.png")

        # data_vis.plot_acceleration_box_and_whisker(x_sph, a_sph, a_sph_pred)

        data_vis.plot_acceleration_residuals(x_sph, a_sph, a_sph_pred, "Spherical Acceleration Residuals", percent=True)
        data_vis.save(plt.gcf(), directory+"Rand_Acc_Comp_Residuals.png")

        # Brillouin Sphere
        x = poly_gm.positions
        a = poly_gm.accelerations
        u = poly_gm.potentials
        data_pred = model.generate_nn_data(x)
        a_pred = data_pred['a']
        u_pred = data_pred['u']
        grid_pot_pred = Grid(trajectory=DH_trajectory, accelerations=format_potential_as_Nx3(u_pred), transform=False)
        grid_acc_pred = Grid(trajectory=DH_trajectory, accelerations=a_pred)

        map_suite.true_pred_acc_pot_plots(grid_pot_true, grid_pot_pred, grid_acc_true, grid_acc_pred, title, percent=True)
        map_vis.save(plt.gcf(), directory+"Brill_Pot_Acc.pdf")
        map_suite.plot_acceleration_comp(grid_acc_pred, title)

        # map_vis.plot_trajectory(train_trajectory)
    plt.show()


if __name__ == "__main__":
    main()
