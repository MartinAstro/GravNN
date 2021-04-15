#import tensorflow_model_optimization as tfmot
from tensorboard.plugins.hparams import api as hp
import multiprocessing as mp
import pandas as pd
import itertools

def main():




    #5,000,000 data --  Plus a little extra epochs for the best performing two 
    df_file = 'Data/Dataframes/hyperparameter_earth_time.data'
    directory = 'logs/hyperparameter_earth_time/'
    hparams = {
        'N_dist' : [5000000],
        'N_train' : [100000, 250000, 500000, 750000, 1000000, 2500000, 4900000],
        'epochs' : [30],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [50000], # the last one is to simulate virtually no decay
        'decay_epoch_0' : [50000],
        'decay_rate' : [2.0],
        'learning_rate' : [2E-2],
        'batch_size': [2**14,2**15,2**16,2**17,2**18,2**19, 2**20],
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'num_units' : [20],
        #'init_file' : [2459314.280798611]
    }

  

    # TODO: Add preprocessing

    keys, values = zip(*hparams.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    args = []
    session_num = 0
    for hparam_inst in permutations_dicts:
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({key: value for key, value in hparam_inst.items()})
        args.append((df_file, directory + run_name, hparam_inst))
        session_num += 1


    # Can't use a pool because the processes get reused, so TF has already been initialized (but apparently not)
    with mp.Pool(1) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()

    for config in configs:
        config = dict(sorted(config.items(), key = lambda kv: kv[0]))
        df = pd.DataFrame().from_dict(config).set_index('timetag')
        try: 
            df_all = pd.read_pickle(df_file)
            df_all = df_all.append(df)
            df_all.to_pickle(df_file)
        except: 
            df.to_pickle(df_file)

def run(df_file, file_name, hparams):
    import tensorflow as tf
    import os

    os.environ["PATH"] += os.pathsep + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"
    os.environ["TF_GPU_THREAD_MODE"] ='gpu_private'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    import copy
    import pickle
    import sys
    import time

    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.io
    from GravNN.Networks import utils

    from GravNN.CelestialBodies.Asteroids import Bennu, Eros
    from GravNN.CelestialBodies.Planets import Earth, Moon
    from GravNN.Networks.Configs.Default_Configs import get_default_moon_config, get_default_earth_config
    from GravNN.Networks.Configs.Fast_Configs import get_fast_earth_config, get_fast_earth_pinn_config
    from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
    from GravNN.GravityModels.SphericalHarmonics import (SphericalHarmonics,
                                                        get_sh_data)
    from GravNN.Networks.Analysis import Analysis
    from GravNN.Networks.Callbacks import CustomCallback, TimingCallback
    from GravNN.Networks.Compression import (cluster_model, prune_model,
                                            quantize_model)
    from GravNN.Networks.Data import generate_dataset, training_validation_split
    from GravNN.Networks.Model import CustomModel
    from GravNN.Networks.Networks import (DenseNet, InceptionNet, ResNet,
                                        TraditionalNet)
    from GravNN.Networks.Plotting import Plotting
    from GravNN.Support.Grid import Grid
    from GravNN.Support.transformations import (cart2sph, project_acceleration,
                                                sphere2cart)

    from GravNN.Trajectories.RandomDist import RandomDist

    from GravNN.Visualization.VisualizationBase import VisualizationBase
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from GravNN.Networks.Activations import bent_identity


    for key, value in hparams.items():
        name = key
        if name == 'decay_rate' : decay_rate = value 
        if name == 'learning_rate': initial_learning_rate = value 
        if name == 'decay_rate_epoch': decay_rate_epochs = value
        if name == 'decay_epoch_0' : decay_epoch_0 = value 
        if name == 'network_shape' : network_shape = value 



    #configurations = {"Default" : get_default_moon_config() }
    #configurations['Default']['init_file'] = [2459304.942048611] # ! Sometimes there is one more sigfig so you might have to check the directory
    #configurations['Default']['N_dist'] =  [5000000]

    configurations = {"Default" : get_default_earth_config() }

    if network_shape == 'normal':
        configurations['Default']['layers'] =  [[3, 20, 20, 20, 20, 20, 20, 20, 20, 3]]
    elif network_shape == 'wide':
        configurations['Default']['layers'] =  [[3, 52, 52, 3]]
    else:
        exit()

    mixed_precision_flag = True

    if sys.platform == 'win32':
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    # # # TODO: Put in mixed precision training
    if mixed_precision_flag:
        from tensorflow.keras.mixed_precision import experimental as mixed_precision

        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print('Compute dtype: %s' % policy.compute_dtype)
        print('Variable dtype: %s' % policy.variable_dtype)


    np.random.seed(1234)
    tf.random.set_seed(0)

    def load_hparams_to_config(hparams, config):
        for key, value in hparams.items():
            config[key] = [value]

        if config['activation'][0] == 'bent_identity':
            config['activation'] = [bent_identity]
        
        try:
            if 'adam' in config['optimizer'][0]:
                config['optimizer'][0] = tf.keras.optimizers.Adam()
        except:
            pass

        try: 
            if 'rmsprop' in config['optimizer'][0]:
                config['optimizer'][0] = tf.keras.optimizers.RMSprop()
        except:
            pass
            
        if 'num_units' in config:
            for i in range(1, len(config['layers'][0])-1):
                config['layers'][0][i] = config['num_units'][0]
        
        if config['network_type'][0] == 'traditional':
            config['network_type'] = [TraditionalNet]
        elif config['network_type'][0] == 'resnet':
            config['network_type'] = [ResNet]
        else:
            exit("Network type (%s) is not defined! Exiting." % config['network_type'][0])

        return config

    def scheduler(epoch, lr):
        epoch0 = decay_epoch_0
        if epoch >= epoch0:
            return initial_learning_rate * (1.0/decay_rate) ** ((epoch-epoch0) / decay_rate_epochs)
        else:
            return lr

    for key, config_original in configurations.items():
        tf.keras.backend.clear_session()
        config = copy.deepcopy(config_original)

        config = load_hparams_to_config(hparams, config)

        utils.check_config_combos(config)
        config = utils.format_config_combos(config)
        print(config)
        
        
        # TODO: Trajectories should take keyword arguments so the inputs dont have to be standard, just pass in config.
        trajectory = config['distribution'][0](config['planet'][0], [config['radius_min'][0], config['radius_max'][0]], config['N_dist'][0], **config)
        if "Planet" in config['planet'][0].__module__:
            get_analytic_data_fcn = get_sh_data
        else:
            get_analytic_data_fcn = get_poly_data
        x_unscaled, a_unscaled, u_unscaled = get_analytic_data_fcn(trajectory, config['grav_file'][0], **config)


        if config['basis'][0] == 'spherical':
            x_unscaled = cart2sph(x_unscaled)     
            a_unscaled = project_acceleration(x_unscaled, a_unscaled)
            x_unscaled[:,1:3] = np.deg2rad(x_unscaled[:,1:3])

        x_train, a_train, u_train, x_val, a_val, u_val = training_validation_split(x_unscaled, 
                                                                                    a_unscaled, 
                                                                                    u_unscaled, 
                                                                                    config['N_train'][0], 
                                                                                    config['N_val'][0])

        a_train = a_train + config['acc_noise'][0]*np.std(a_train)*np.random.randn(a_train.shape[0], a_train.shape[1])


        # Preprocessing
        x_transformer = config['x_transformer'][0]
        a_transformer = config['a_transformer'][0]
        u_transformer = config['u_transformer'][0]

        x_train = x_transformer.fit_transform(x_train)
        a_train = a_transformer.fit_transform(a_train)
        u_train = u_transformer.fit_transform(u_train)

        x_val = x_transformer.transform(x_val)
        a_val = a_transformer.transform(a_val)
        u_val = u_transformer.transform(u_val)

        # Decide to train with potential or not
        y_train = np.hstack([u_train, a_train]) if config['use_potential'][0] else np.hstack([np.zeros(np.shape(u_train)), a_train])
        y_val = np.hstack([u_val, a_val]) if config['use_potential'][0] else np.hstack([np.zeros(np.shape(u_val)), a_val])


        dataset = generate_dataset(x_train, y_train, config['batch_size'][0])
        val_dataset = generate_dataset(x_val, y_val, config['batch_size'][0])

        if config['init_file'][0] is not None:
            network = tf.keras.models.load_model(os.path.abspath('.') +"/Data/Networks/"+str(config['init_file'][0])+"/network")
        else:
            network = config['network_type'][0](**config)        
        
        model = CustomModel(config, network)

        callback = TimingCallback()
        schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
        lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5000, factor=0.5, min_delta=0.00001, verbose=1, min_lr=1E-5)

        optimizer = config['optimizer'][0]
        optimizer.learning_rate = config['learning_rate'][0]
        if config['mixed_precision'][0]:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
        else:
            optimizer.get_scaled_loss = lambda x: x
            optimizer.get_unscaled_gradients = lambda x: x
        model.compile(optimizer=optimizer, loss="mse")#, run_eagerly=True)#, metrics=["mae"])

        history = model.fit(dataset, 
                            epochs=config['epochs'][0], 
                            verbose=0,
                            validation_data=val_dataset,
                            callbacks=[callback, schedule])#lr_on_plateau])# schedule])#, hyper_params])# tensorboard, hyper_params])#,
                                        #early_stop])
        history.history['time_delta'] = callback.time_delta
        model.config['delta'] = callback.time_10
        model.history = history

        # TODO: Save extra parameters like optimizer.learning_rate
        # Save network and config information
        model.config['time_delta'] = [callback.time_delta]
        model.config['x_transformer'][0] = x_transformer
        model.config['a_transformer'][0] = a_transformer


        #model.save(df_file)
        # Need to make this async friendly on the dataframe
        timestamp = pd.Timestamp(time.time(), unit='s').round('s').ctime()
        model.directory = os.path.abspath('.') +"/Data/Networks/"+ str(pd.Timestamp(timestamp).to_julian_date()) + "/"
        os.makedirs(model.directory, exist_ok=True)
        model.network.save(model.directory + "network")
        model.config['timetag'] = timestamp
        model.config['history'] = [model.history.history]
        model.config['id'] = [pd.Timestamp(timestamp).to_julian_date()]
        try:
            model.config['activation'] = [model.config['activation'][0].__name__]
        except:
            pass
        try:
            model.config['optimizer'] = [model.config['optimizer'][0].__module__]
        except:
            pass
        model.model_size_stats()

        config = dict(sorted(config.items(), key = lambda kv: kv[0]))
        df = pd.DataFrame().from_dict(config).set_index('timetag')
        df.to_pickle(model.directory + "config.data")

        return model.config



if __name__ == '__main__':
    main()
