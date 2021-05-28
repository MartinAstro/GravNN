import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import (SphericalHarmonics,
                                                     get_sh_data)
from GravNN.Networks import utils
from GravNN.Networks.Constraints import no_pinn, pinn_A
from GravNN.Networks.Model import CustomModel, load_config_and_model
from GravNN.Trajectories import DHGridDist
from GravNN.Visualization.MapVisualization import MapVisualization

if sys.platform == 'win32':
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def generate_data(traj,config):
    # Generate the plotting data 
    x, a, u = get_sh_data(traj, config['grav_file'][0], **config)
    x_transformer = config['x_transformer'][0]
    a_transformer = config['a_transformer'][0]
    x = x_transformer.transform(x)
    return x

def generate_layer_models(model):
    models = []
    for layer_i in range(0, len(model.network.layers)):
        new_model = tf.keras.Model(model.network.inputs, model.network.layers[layer_i].output)
        models.append(new_model)
    
    return models

def plot_intermediate_layer(config, model, columns, planet=None):
    mapUnit = 'mGal'
    map_vis = MapVisualization(mapUnit)
    #map_vis.save_directory = os.path.abspath('.') +"/Plots/"
    map_vis.fig_size = map_vis.full_page
    plt.rc('text', usetex=False)
    map_vis.newFig()

    if planet is None:
        planet = Earth()

    density_deg = 180
    traj = DHGridDist(planet, planet.radius, degree=density_deg)

    x = generate_data(traj, config)
    N_layers = len(model.network.layers)
    layer_models = generate_layer_models(model)

    # Format the number of images to be shown on the plot
    rows = N_layers # a row per layer
    cols = columns # number of random nodes to sample

    row_i = 0
    for layer_model in layer_models:
        basis_functions = np.array(layer_model(x), dtype=np.float32)
        col_i = 0
        for k in range(cols):
            if k >= basis_functions.shape[1]:
                continue
            basis_function = np.transpose(basis_functions)[k]


            # If it is the first or last layer, make sure the 3 components are centered
            if (row_i == 0 or row_i == N_layers-1) and col_i == 0:
                col_i = cols//2 - 1

            plt.subplot(rows, cols, row_i*cols+col_i+1)

            grid_pred = np.reshape(basis_function,(traj.N_lon,traj.N_lat))
            sigma=1
            im = map_vis.new_map(grid_pred)#, vlim=[np.mean(grid_pred) - sigma*np.std(grid_pred), np.mean(grid_pred) + sigma*np.std(grid_pred)])
            im.set_clim(vmin=(np.mean(grid_pred) - sigma*np.std(grid_pred))*10000,vmax=(np.mean(grid_pred) + sigma*np.std(grid_pred))*10000)
            plt.xlabel(None)
            plt.ylabel(None)
            plt.xticks([])
            plt.yticks([])
            col_i += 1
        row_i += 1
   
    map_vis.save(plt.gcf(), "Intermediate_Layers.pdf")
    
def load_config_and_model(model_id, df_file):
    # Get the parameters and stats for a given run
    # If the dataframe hasn't been loaded
    if type(df_file) == str:
        config = utils.get_df_row(model_id, df_file)
    else:
        # If the dataframe has already been loaded
        config = df_file[model_id == df_file['id']].to_dict()
        for key, value in config.items():
            config[key] = list(value.values())

    if 'use_potential' not in config:
        config['use_potential'] = [False]
    if 'mixed_precision' not in config:
        config['use_precision'] = [False]
    if 'PINN_constraint_fcn' not in config:
        config['PINN_constraint_fcn'] = [no_pinn]
    if 'dtype' not in config:
        config['dtype'] = [tf.float32]
    if 'dtype' not in config:
        config['dtype'] = [tf.float32]
    if 'class_weight' not in config:
        config['class_weight'] = [1.0]
    # Reinitialize the model
    network = tf.keras.models.load_model('C:\\Users\\John\\Documents\\Research\\ML_Gravity' + "/Data/Networks/"+str(model_id)+"/network")
    model = CustomModel(config, network)
    optimizer = utils._get_optimizer(config['optimizer'][0])
    model.compile(optimizer=optimizer, loss='mse') #! Check that this compile is even necessary

    return config, model

def plot_intermediate_layers_helper(idx, df, columns, planet=None):
    ids = df['id'].values
    model_id = ids[idx]
    config, model = load_config_and_model(model_id, df)
    plot_intermediate_layer(config, model, columns, planet)

def main():
    df_file = 'C:\\Users\\John\\Documents\\Research\\ML_Gravity\\Data\\Dataframes\\traditional_nn_df.data'
    bent_df = pd.read_pickle(df_file)#.sort_values(by='Brillouin_rse_mean', ascending=True)
    plot_intermediate_layers_helper(1, bent_df, 5, Earth())
    plt.show()

if __name__ == "__main__":
    main()
