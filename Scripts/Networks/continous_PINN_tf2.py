
import os

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"

import copy
import pickle
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Networks import utils
from GravNN.Networks.Analysis import Analysis
from GravNN.Networks.Callbacks import CustomCallback
from GravNN.Networks.Compression import (cluster_model, prune_model,
                                         quantize_model)
from GravNN.Networks.Data import training_validation_split
from GravNN.Networks.Model import CustomModel
from GravNN.Networks.Networks import (DenseNet, InceptionNet, ResNet,
                                      TraditionalNet)
from GravNN.Networks.Plotting import Plotting
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Trajectories.ReducedRandDist import ReducedRandDist
from GravNN.Support.Grid import Grid
from GravNN.Support.transformations import cart2sph, sphere2cart, project_acceleration
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from sklearn.preprocessing import MinMaxScaler, StandardScaler

np.random.seed(1234)
tf.random.set_seed(0)

def main():
    planet = Earth()
    gravity_file = planet.sh_hf_file
    density_deg = 180
    save = True
    train = True
    df_file = 'temp.data'
    df_file = 'param_study.data'
    #df_file = "tensorflow_2_results.data"
    #df_file = "tensorflow_2_ablation.data"
    
    inception_layer = [3, 7, 11]
    dense_layer = [10, 10, 10]
    data_config = {
        'distribution' : [RandomDist],
        'N_train' : [250000], 
        'N_val' : [4000],
        'radius_min' : [planet.radius],
        'radius_max' : [planet.radius + 10.0],
        'acc_noise' : [0.00],
        'basis' : [None],# ['spherical'],
        'deg_removed' : [2],
        'include_U' : [False],
        'max_deg' : [1000], 
        'sh_truth' : ['sh_stats_']
    }
    compression_config = {
        'fine_tuning_epochs' : [25000],
        'sparsity' : [None], # None, 0.5, 0.8
        'num_w_clusters' : [None], # None, 32, 16
        'quantization' : [None] 
    }
    network_config = {
        'preprocessing' : [MinMaxScaler],
        'network_type' : [TraditionalNet],
        'PINN_flag' : [False],
        'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
        'activation' : ['tanh'],
        'init_file' : [None],#'2459192.4530671295'],
        'epochs' : [100000],
        'optimizer' : [tf.keras.optimizers.Adam],
        'batch_size' : [160000],
        'dropout' : [0.0]
    }
    
    # ResNet -- 'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
    # DenseNet -- 'layers' : [[3, dense_layer, [10], dense_layer, [10], dense_layer, [10], dense_layer, 3]],
    # InceptionNet -- 'layers' : [[3, inception_layer, inception_layer, inception_layer, inception_layer, 1]],


    config = {}
    config.update(data_config)
    config.update(network_config)
    config.update(compression_config)

    # Dropout

    
    config_1 = copy.deepcopy(config)
    config_1.update({'layers' : [[3, 80, 80, 80, 80, 80, 80, 80, 80, 3]],
                    'PINN_flag' : [False],
                    'N_train' : [950000], 
                     'batch_size' : [40000],
                     'epochs' : [100000],
                     'basis' : [None],
                     'radius_min' : [planet.radius + 420000-10],
                     'radius_max' : [planet.radius + 420000+10],
                    })

    
    config_2 = copy.deepcopy(config)
    config_2.update({'layers' : [[3, 80, 80, 80, 80, 80, 80, 80, 80, 3]],
                    'PINN_flag' : [False],
                     'batch_size' : [40000],
                     'N_train' : [950000], 
                     'epochs' : [100000],
                     'basis' : [None],
                     'radius_min' : [planet.radius],
                     'radius_max' : [planet.radius+10],
                    })

    config_3 = copy.deepcopy(config)
    config_3.update({'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
                    'PINN_flag' : [False],
                    'N_train' : [950000], 
                     'batch_size' : [40000],
                     'epochs' : [100000],
                     'basis' : [None],
                     'radius_min' : [planet.radius + 420000-10],
                     'radius_max' : [planet.radius + 420000+10],
                    })

    
    config_4 = copy.deepcopy(config)
    config_4.update({'layers' : [[3, 40, 40, 40, 40, 40, 40, 40, 40, 3]],
                    'PINN_flag' : [False],
                     'batch_size' : [40000],
                     'N_train' : [950000], 
                     'epochs' : [100000],
                     'basis' : [None],
                     'radius_min' : [planet.radius],
                     'radius_max' : [planet.radius+10],
                    })



    configurations = {
         "1" : config_1,
         "2" : config_2,
         "3" : config_3,
         "4" : config_4
    }    

    for key, config in configurations.items():
        tf.keras.backend.clear_session()

        utils.check_config_combos(config)
        
        trajectory = config['distribution'][0](planet, [config['radius_min'][0], config['radius_max'][0]], points=1000000)
        x_unscaled, a_unscaled, u_unscaled = get_sh_data(trajectory, planet, gravity_file,config['max_deg'][0], config['deg_removed'][0])

        if config['basis'][0] == 'spherical':
            x_unscaled = cart2sph(x_unscaled)     
            a_unscaled = project_acceleration(x_unscaled, a_unscaled)
            x_unscaled[:,1:3] = np.deg2rad(x_unscaled[:,1:3])
            # non_divergent_idx = (x_unscaled[:,2] != 0 or x_unscaled[:,2] != np.deg2rad(180.0))
            # x_unscaled = x_unscaled[non_divergent_idx] 
            # a_unscaled = a_unscaled[non_divergent_idx]

        x_train, a_train, u_train, x_val, a_val, u_val = training_validation_split(x_unscaled, 
                                                                                    a_unscaled, 
                                                                                    u_unscaled, 
                                                                                    config['N_train'][0], 
                                                                                    config['N_val'][0])


        # Preprocessing
        try:
            x_transformer = config['preprocessing'][0](feature_range=(-1,1))
            a_transformer = config['preprocessing'][0](feature_range=(-1,1))
        except:
            x_transformer = config['preprocessing'][0]()
            a_transformer = config['preprocessing'][0]()

        x_train = x_transformer.fit_transform(x_train)
        a_train = a_transformer.fit_transform(a_train)

        x_val = x_transformer.transform(x_val)
        a_val = a_transformer.transform(a_val)
        
        # Initial Data

        # Add Noise if interested
        a_train = a_train + config['acc_noise'][0]*np.std(a_train)*np.random.randn(a_train.shape[0], a_train.shape[1])

        x_train = x_train.astype('float32')
        a_train = a_train.astype('float32')

        x_val = x_val.astype('float32')
        a_val = a_val.astype('float32')

        dataset = tf.data.Dataset.from_tensor_slices((x_train, a_train))
        dataset = dataset.shuffle(1000, seed=1234)
        dataset = dataset.batch(config['batch_size'][0])
        dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, a_val))
        val_dataset = val_dataset.shuffle(1000, seed=1234)
        val_dataset = val_dataset.batch(config['batch_size'][0])
        val_dataset = val_dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
        val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.cache()
        #Why Cache is Impt: https://stackoverflow.com/questions/48240573/why-is-tensorflows-tf-data-dataset-shuffle-so-slow

        network = config['network_type'][0](config['layers'][0], config['activation'][0], dropout=config['dropout'][0])
        model = CustomModel(config, network)
        test_trajectories = {
            "Brillouin" : DHGridDist(planet, planet.radius, degree=density_deg),
            #"LEO" : DHGridDist(planet, planet.radius+420000.0, degree=density_deg),
            #"GEO" : DHGridDist(planet, planet.radius+35786000.0, degree=density_deg)
        }
        analyzer = Analysis(gravity_file, test_trajectories, x_transformer, a_transformer, config)

        ######################################################################
        ############################# Train Model    #########################
        ######################################################################    

        if config['init_file'][0] is None:
            callback = CustomCallback()
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=1E-5, patience=1000, verbose=0,
                                            mode='auto', baseline=None, restore_best_weights=True
                                        )
            try:
                optimizer = config['optimizer'][0](learning_rate=config['lr_scheduler'][0])
            except:
                print("EXCEPTION INITIALIZING OPTIMIZER")
                optimizer = config['optimizer'][0]()

            model.compile(optimizer=optimizer, loss="mse")#, run_eagerly=True)#, metrics=["mae"])
            # dot_img_file = 'model_1.png'
            # tf.keras.utils.plot_model(model.network, to_file=dot_img_file, show_shapes=True)
            history = model.fit(dataset, 
                                epochs=config['epochs'][0], 
                                verbose=0,
                                validation_data=val_dataset,
                                callbacks=[callback])#,
                                            #early_stop])
            history.history['time_delta'] = callback.time_delta
            #model.optimize(dataset)

            analyzer(model, history, df_file, save)  
        else:
            network = tf.keras.models.load_model(os.path.abspath('.') +"/Plots/"+str(config['init_file'][0])+"/network")
            model = CustomModel(config, network)
            model.compile(optimizer='adam', loss='mse')
            history = None
            time_delta = 0

        ######################################################################
        ############################# Optimize Model #########################
        ######################################################################    

        # model, cluster_history = cluster_model(model, dataset, config)
        # analyzer(model, cluster_history, df_file, save)

        # model, prune_history = prune_model(model, dataset, config)
        # analyzer(model, prune_history, df_file, save)

        #model, quantize_history = quantize_model(model, dataset, config)
        print("mean SH: " + str(analyzer.stats['Brillouin_sh_diff_mean']))
        print("median SH: " + str(analyzer.stats['Brillouin_sh_diff_median']))
        print("2 median SH: " + str(analyzer.stats['Brillouin_sh_sigma_2_median']))
        print("2c median SH: " + str(analyzer.stats['Brillouin_sh_sigma_2_c_median']))

        ######################################################################
        ############################# Plotting ###############################
        ######################################################################    

        plotter = Plotting(gravity_file, test_trajectories, x_transformer, a_transformer, config, analyzer.stats['directory'][0])
        plotter.plot_maps(model)
        plotter.plot_history([history], ['vanilla'])

        if not save:            
            plt.show()

        plt.close()



if __name__ == '__main__':
    main()
