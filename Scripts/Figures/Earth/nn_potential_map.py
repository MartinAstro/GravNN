        
import os
import pickle
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Grid import Grid
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.Data import standardize_output
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase

def format_potential_as_Nx3(model):
    Clm_p = model.potential
    Clm_a = model.acceleration
    Clm_p_3 = np.zeros(np.shape(Clm_a))
    Clm_p_3[:,0] = Clm_p
    return Clm_p_3

def get_potential(x,model):
    new_model = tf.keras.Model(model.network.inputs, model.network.layers[-1].output)
    u = new_model(x)
    return u


def generate_nn_data(x, a, model, config):
    if config['basis'][0] == 'spherical':
        x = cart2sph(x)
        x[:,1:3] = np.deg2rad(x[:,1:3])

    x_transformer = config['x_transformer'][0]
    a_transformer = config['a_transformer'][0]
    u_transformer = config['u_transformer'][0]
    x = x_transformer.transform(x)
    u_pred = get_potential(x, model)
    u_pred = u_transformer.inverse_transform(u_pred)
    u_3vec = np.zeros(np.shape(a))
    u_3vec[:,0] = u_pred[:,0]

    y_hat = model.predict(x)
    u_pred, a_pred, laplace_pred, curl_pred = standardize_output(y_hat, config)
    a_pred = a_transformer.inverse_transform(a_pred)

    return u_3vec, a_pred

def main():
    
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180 # 180
    max_deg = 1000

    radius_min = planet.radius
    trajectory = DHGridDist(planet, radius_min, degree=density_deg)
    Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory).load()


    df = pd.read_pickle('Data/Dataframes/hyperparameter_earth_pinn_20_v10.data')
    model_id = df['id'].values[-1]
    config, model = load_config_and_model(model_id, df)

    u, a_pred = generate_nn_data(trajectory.positions, Call_r0_gm.accelerations, model, config)
    
    grid_true = Grid(trajectory=trajectory, accelerations=u)
   
    directory = os.path.abspath('.') +"/Plots/OneOff/"
    os.makedirs(directory, exist_ok=True)

    mapUnit = 'm/s^2'
    map_vis = MapVisualization(mapUnit)
    map_vis.fig_size = map_vis.full_page
    map_vis.plot_grid(grid_true.r, "Potential")

    grid_acceleration = Grid(trajectory=trajectory, accelerations=a_pred)
    map_vis.plot_grid(grid_acceleration.total, "Acceleration")
    plt.show()

if __name__ == "__main__":
    main()
