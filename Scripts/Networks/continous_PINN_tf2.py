
import os
os.environ["PATH"] += os.pathsep + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"

import pickle
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import tensorflow as tf

from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Trajectories.ReducedRandDist import ReducedRandDist
from GravNN.Visualization.Grid import Grid
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.Visualization.VisualizationBase import VisualizationBase
from sklearn.preprocessing import MinMaxScaler
from GravNN.Networks.model import PhysicsInformedNN
from scipy.optimize import minimize, fmin_l_bfgs_b
import tensorflow_model_optimization as tfmot

np.random.seed(1234)
#tf.compat.v1.disable_eager_execution()

def main():
    planet = Earth()
    model_file = planet.sh_hf_file
    density_deg = 180
    max_deg = 1000
    save = False
    train = True
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    data_config = {
        'distribution' : [RandomDist],
        'N_train' : [40000], 
        'radius_max' : [planet.radius + 10.0],
        'acc_noise' : [0.00],
        'preprocessing' : [MinMaxScaler],
        'deg_removed' : [2],
        'include_U' : [False]
    }
    network_config = {
        'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]],
        'activation' : ['tanh'],
        'init_model' : [None],
        'epochs' : [200000],
        'optimizer' : [tf.keras.optimizers.Adam],
        'PINN_flag' : [False],
        'batch_size' : [4096]
    }
    configurations = {
        "config_nonPINN" : {
            'N_train' : [40000],
            'PINN_flag' : [True],
            'epochs' : [200000], 
            'radius_max' : [planet.radius + 10.0],
            'layers' : [[3, 20, 20, 20, 20, 20, 20, 20, 20, 1]],
            'acc_noise' : [0.00],
            'deg_removed' : [2],
            'activation' : ['tanh'],
            'init_file': [None],
            'notes' : ['nonPINN - No potential included'],
            'batch_size' : [8192]#96]#[8192]#4096]#4096]
        },
    }    

    for key, config in configurations.items():
        tf.keras.backend.clear_session()
        #tf.debugging.set_log_device_placement(True)
        #tf.reset_default_graph()

        radius_min = planet.radius

        df_file = "continuous_results.data"

        trajectory = RandomDist(planet, [radius_min, config['radius_max'][0]], points=259200)
        map_trajectory =  DHGridDist(planet, radius_min, degree=density_deg)

        Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=trajectory)
        accelerations = Call_r0_gm.load()

        Clm_r0_gm = SphericalHarmonics(model_file, degree=int(config['deg_removed'][0]), trajectory=trajectory)
        accelerations_Clm = Clm_r0_gm.load()

        x_unscaled = Call_r0_gm.positions # position (N x 3)
        a_unscaled = accelerations - accelerations_Clm
        u = None # potential (N x 1)

        # Preprocessing
        x_transformer = MinMaxScaler(feature_range=(-1,1))
        x = x_transformer.fit_transform(x_unscaled)

        a_transformer = MinMaxScaler(feature_range=(-1,1))
        a = a_transformer.fit_transform(a_unscaled)
        
        # Initial Data
        idx_x = np.random.choice(x.shape[0], config['N_train'][0], replace=False) 

        x_train = x[idx_x] # (x,y,z) or (r, theta, phi)
        a_train = a[idx_x] # (a_x, a_y, a_z) or (a_r, a_theta, a_phi)

        # Add Noise if interested
        a_train = a_train + config['acc_noise'][0]*np.std(a_train)*np.random.randn(a_train.shape[0], a_train.shape[1])

        x_train = x_train.astype('float32')
        a_train = a_train.astype('float32')

        PINN = PhysicsInformedNN(config)

        dataset = tf.data.Dataset.from_tensor_slices((x_train, a_train))
        dataset = dataset.shuffle(1000)
        dataset = dataset.batch(config['batch_size'][0])
        dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()

        #Why Cache is Impt: https://stackoverflow.com/questions/48240573/why-is-tensorflows-tf-data-dataset-shuffle-so-slow
        #dataset = dataset.shuffle(1000, seed=1234)
        #dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


        start = time.time()
        if train:
            PINN.train(dataset=dataset,
                       epochs=config['epochs'][0],
                       batch_size=None)
        time_delta = np.round(time.time() - start, 2)

        ######################################################################
        ############################# Prune Model    #########################
        ######################################################################    

        if False:
            #Pruning https://blog.tensorflow.org/2019/05/tf-model-optimization-toolkit-pruning-API.html
            pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                                    initial_sparsity=0.0, final_sparsity=0.5,
                                    begin_step=2000, end_step=4000)

            model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)
            PINN.network = model_for_pruning
            PINN.train(dataset=dataset,
                        epochs=config['fine_tune_epochs'][0],
                        batch_size=None)

            #Weight Clustering: https://blog.tensorflow.org/2020/08/tensorflow-model-optimization-toolkit-weight-clustering-api.html
            clustering_params = {
                'number_of_clusters': 32,
                'cluster_centroids_init': tfmot.clustering.keras.CentroidInitialization.LINEAR
            }

            clustered_model = cluster_weights(PINN.network, **clustering_params)
            PINN.network = clustered_model
            PINN.train(dataset=dataset,
                        epochs=config['fine_tune_epochs'][0],
                        batch_size=None)
            model_for_serving = tfmot.clustering.keras.strip_clustering(clustered_model)

            # Quantizations: https://blog.tensorflow.org/2020/04/quantization-aware-training-with-tensorflow-model-optimization-toolkit.html
            #quantized_model = tfmot.quantization.keras.quantize_model(model)


        ######################################################################
        ############################# Training Stats #########################
        ######################################################################    

        Call_r0_gm = SphericalHarmonics(model_file, degree=max_deg, trajectory=map_trajectory)
        Call_a = Call_r0_gm.load()
        
        Clm_r0_gm = SphericalHarmonics(model_file, degree=int(config['deg_removed'][0]), trajectory=map_trajectory)
        Clm_a = Clm_r0_gm.load()

        x = Call_r0_gm.positions # position (N x 3)
        a = Call_a - Clm_a

        x = x_transformer.transform(x)
        a = a_transformer.transform(a)

        x_pred = tf.data.Dataset.from_tensors((x.astype('float32')))

        U_pred, acc_pred = PINN.predict(x_pred)

        x = x_transformer.inverse_transform(x)
        a = a_transformer.inverse_transform(a)
        acc_pred = a_transformer.inverse_transform(acc_pred)

        error = np.abs(np.divide((acc_pred - a), a))*100 # Percent Error for each component
        RSE_Call = np.sqrt(np.square(acc_pred - a))

        params = np.sum([np.prod(v.get_shape().as_list()) for v in PINN.network.trainable_variables])
        timestamp = pd.Timestamp(time.time(), unit='s').round('s').ctime()
        entries = {
            'timetag' : [timestamp],
            'trajectory' : [trajectory.__class__.__name__],
            'radius_min' : [radius_min],
            'train_time' : [time_delta],
            'degree' : [max_deg],
            'params' : [params]

        }
        training_stats = {
            'rse_mean' : [np.mean(RSE_Call)],
            'rse_std' : [np.std(RSE_Call)],
            'rse_median' : [np.median(RSE_Call)],
            'rse_a0_mean' : [np.mean(RSE_Call[:,0])],
            'rse_a1_mean' : [np.mean(RSE_Call[:,1])],
            'rse_a2_mean' : [np.mean(RSE_Call[:,2])],

            'percent_rel_mean' : [np.mean(error)],
            'percent_rel_std' : [np.std(error)], 
            'percent_rel_median' : [np.median(error)],
            'percent_rel_a0_mean' : [np.mean(error[:,0])], 
            'percent_rel_a1_mean' : [np.mean(error[:,1])], 
            'percent_rel_a2_mean' : [np.mean(error[:,2])],

        }
        config.update(training_stats)
        config.update(entries)

        ######################################################################
        ############################# Testing Stats ##########################
        ######################################################################    

        grid_true = Grid(trajectory=map_trajectory, accelerations=a)
        grid_pred = Grid(trajectory=map_trajectory, accelerations=acc_pred)
        diff = grid_pred - grid_true
       
        # This ensures the same features are being evaluated independent of what degree is taken off at beginning
        C22_r0_gm = SphericalHarmonics(model_file, degree=2, trajectory=map_trajectory)
        C22_a = C22_r0_gm.load()
        grid_C22 = Grid(trajectory=map_trajectory, accelerations=Call_a - C22_a)

        two_sigma_mask = np.where(grid_C22.total > (np.mean(grid_C22.total) + 2*np.std(grid_C22.total)))
        two_sigma_mask_compliment = np.where(grid_C22.total < (np.mean(grid_C22.total) + 2*np.std(grid_C22.total)))
        two_sig_features = diff.total[two_sigma_mask]
        two_sig_features_comp = diff.total[two_sigma_mask_compliment]

        three_sigma_mask = np.where(grid_C22.total > (np.mean(grid_C22.total) + 3*np.std(grid_C22.total)))
        three_sigma_mask_compliment = np.where(grid_C22.total < (np.mean(grid_C22.total) + 3*np.std(grid_C22.total)))
        three_sig_features = diff.total[three_sigma_mask]
        three_sig_features_comp = diff.total[three_sigma_mask_compliment]

        map_stats = {
            'sigma_2_mean' : [np.mean(np.sqrt(np.square(two_sig_features)))],
            'sigma_2_std' : [np.std(np.sqrt(np.square(two_sig_features)))],
            'sigma_2_median' : [np.median(np.sqrt(np.square(two_sig_features)))],

            'sigma_2_c_mean' : [np.mean(np.sqrt(np.square(two_sig_features_comp)))],
            'sigma_2_c_std' : [np.std(np.sqrt(np.square(two_sig_features_comp)))],
            'sigma_2_c_median' : [np.median(np.sqrt(np.square(two_sig_features_comp)))],

            'sigma_3_mean' : [np.mean(np.sqrt(np.square(three_sig_features)))],
            'sigma_3_std' : [np.std(np.sqrt(np.square(three_sig_features)))],
            'sigma_3_median' : [np.median(np.sqrt(np.square(three_sig_features)))],

            'sigma_3_c_mean' : [np.mean(np.sqrt(np.square(three_sig_features_comp)))],
            'sigma_3_c_std' : [np.std(np.sqrt(np.square(three_sig_features_comp)))],
            'sigma_3_c_median' : [np.median(np.sqrt(np.square(three_sig_features_comp)))],

            'max_error' : [np.max(np.sqrt(np.square(diff.total)))]
        }
        config.update(map_stats)
        df = pd.DataFrame().from_dict(config).set_index('timetag')

        ######################################################################
        ############################# Plotting ###############################
        ######################################################################    

        mapUnit = 'mGal'
        map_vis = MapVisualization(mapUnit)
        plt.rc('text', usetex=False)

        fig_true, ax = map_vis.plot_grid(grid_true.total, "True Grid [mGal]")
        fig_pred, ax = map_vis.plot_grid(grid_pred.total, "NN Grid [mGal]")
        fig_pert, ax = map_vis.plot_grid(diff.total, "Acceleration Difference [mGal]")

        map_vis.fig_size = (5*4,3.5*4)
        fig, ax = map_vis.newFig()
        vlim = [0, np.max(grid_true.total)*10000.0] 
        plt.subplot(311)
        im = map_vis.new_map(grid_true.total, vlim=vlim, log_scale=False)
        map_vis.add_colorbar(im, '[mGal]', vlim)
        
        plt.subplot(312)
        im = map_vis.new_map(grid_pred.total, vlim=vlim, log_scale=False)
        map_vis.add_colorbar(im, '[mGal]', vlim)
        
        plt.subplot(313)
        im = map_vis.new_map(diff.total, vlim=vlim, log_scale=False)
        map_vis.add_colorbar(im, '[mGal]', vlim)

        ######################################################################
        ############################# Saving #################################
        ######################################################################    

        if save: 
            try: 
                df_all = pd.read_pickle(df_file)
                df_all = df_all.append(df)
                df_all.to_pickle(df_file)
            except: 
                df.to_pickle(df_file)

            directory = os.path.abspath('.') +"/Plots/"+ str(pd.Timestamp(timestamp).to_julian_date()) + "/"
            os.makedirs(directory, exist_ok=True)

            map_vis.save(fig_true, directory + "true.pdf")
            map_vis.save(fig_pred, directory + "pred.pdf")
            map_vis.save(fig_pert, directory + "diff.pdf")
            map_vis.save(fig, directory + "all.pdf")

            with open(directory + "network.data", 'wb') as f:
                weights = []
                biases = []
                for i in range(len(model.weights)):
                    weights.append(model.weights[i].eval(session=model.sess))
                    biases.append(model.biases[i].eval(session=model.sess))
                pickle.dump(weights, f)
                pickle.dump(biases, f)
        else:
            plt.show()

        plt.close()
        #model.sess.close()

if __name__ == '__main__':
    main()