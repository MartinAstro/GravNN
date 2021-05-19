#import tensorflow_model_optimization as tfmot
from tensorboard.plugins.hparams import api as hp
import multiprocessing as mp
import pandas as pd
import itertools
from GravNN.Trajectories.ExponentialDist import ExponentialDist

def main():

    # Take trained network and see if a larger learning rate would have let it continue learning. 
    df_file = 'Data/Dataframes/hyperparameter_moon_pinn_20_v10.data'
    directory = 'logs/hyperparameter_moon_pinn_20_v10/'

    df_file = 'Data/Dataframes/exponential_invert_dist_v2.data'
    directory = 'logs/exponential_invert_dist_v2/'

    # df_file = 'Data/Dataframes/hyperparameter_earth_20_v20.data'
    # directory = 'logs/hyperparameter_earth_20_v20/'

    df_file = 'Data/Dataframes/exponential_invert_dist_v10.data'
    directory = 'logs/exponential_invert_dist_v10/'

    df_file = 'Data/Dataframes/hyperparameter_moon_pinn_80_v10.data'
    directory = 'logs/hyperparameter_moon_pinn_80_v10/'

    df_file = 'Data/Dataframes/hyperparameter_moon_traditional.data'
    directory = 'logs/hyperparameter_moon_traditional/'

    df_file = 'Data/Dataframes/hyperparameter_moon_pinn_40_v10.data'
    directory = 'logs/hyperparameter_moon_pinn_40_v10/'

    threads = 6
    threads = 1

    hparams = {
        'N_dist' : [5000000],#[1200000],
        'N_train' :[4900000],#[1000000], #
        'epochs' : [100000],
        'network_shape' : ['normal'],
        'decay_rate_epoch' : [25000],
        'decay_epoch_0' : [25000],
        'decay_rate' : [2.0],
        'learning_rate' : [0.005], # 5e-3 (highest), 5e-3*(1/2)^1=0.0025, (5e-3)*(1/2)^(2)=0.00125 (lowest) 
        'batch_size': [131072*2],
        'activation' : ['gelu'],
        'initializer' : ['glorot_uniform'],
        'network_type' : ['traditional'],
        'PINN_constraint_fcn' :['pinn_A'],#,# ['no_pinn'],#
        # 'x_transformer' : [MinMaxScaler(feature_range=(-1,1))],
        # 'u_transformer' : [UniformScaler(feature_range=(-1,1))],
        # 'a_transformer' : [UniformScaler(feature_range=(-1,1))],
        'scale_by' : ['a'],
        'mixed_precision' : [False],
        'num_units' : [40],#, 80],#, 80], 
        # 'distribution' : [ExponentialDist],
        # 'invert' : [True],
        # 'scale_parameter' : [420000.0/10.0]
    }



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
    with mp.Pool(threads) as pool:
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
    from GravNN.Networks.Configs.Default_Configs import get_default_moon_config, get_default_earth_config, \
                                                        get_default_earth_pinn_config, get_default_moon_pinn_config
    from GravNN.Networks.Configs.Fast_Configs import get_fast_earth_config, get_fast_earth_pinn_config
    from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
    from GravNN.GravityModels.SphericalHarmonics import (SphericalHarmonics,
                                                        get_sh_data)
    from GravNN.Networks.Analysis import Analysis
    from GravNN.Networks.Callbacks import CustomCallback
    from GravNN.Networks.Compression import (cluster_model, prune_model,
                                            quantize_model)
    from GravNN.Networks.Data import generate_dataset, training_validation_split, get_preprocessed_data, configure_dataset
    from GravNN.Networks.Model import CustomModel
    from GravNN.Networks.Networks import (DenseNet, InceptionNet, ResNet,
                                        TraditionalNet, load_network)
    from GravNN.Networks.Plotting import Plotting
    from GravNN.Networks.Constraints import no_pinn, pinn_A
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
        if name == 'mixed_precision' : mixed_precision_flag = value
        if name == 'PINN_constraint_fcn' : PINN_constraint_fcn_val = value



    #configurations = {"Default" : get_default_moon_config() }
    #configurations['Default']['init_file'] = [2459304.942048611] # ! Sometimes there is one more sigfig so you might have to check the directory
    #configurations['Default']['N_dist'] =  [5000000]

    if PINN_constraint_fcn_val == 'no_pinn':
        #configurations = {"Default" : get_default_earth_config() }
        configurations = {"Default" : get_default_moon_config() }
    else:
        #configurations = {"Default" : get_default_earth_pinn_config() }
        configurations = {"Default" : get_default_moon_pinn_config() }

    #configurations = {"Default" : get_default_moon_pinn_config() }

    if network_shape == 'normal':
        configurations['Default']['layers'] =  [[3, 20, 20, 20, 20, 20, 20, 20, 20, 1]]
    elif network_shape == 'wide':
        configurations['Default']['layers'] =  [[3, 52, 52, 1]]
    else:
        exit()

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

        # try:
        #     config['PINN_constraint_fcn'] = [eval(config['PINN_constraint_fcn'][0])]
        # except:
        #     exit("Couldn't load the constraint!")

        if config['PINN_constraint_fcn'][0] == 'pinn_A':
            config['PINN_constraint_fcn'] = [pinn_A]
        elif config['PINN_constraint_fcn'][0] == 'no_pinn':
            config['PINN_constraint_fcn'] = [no_pinn]
        else:
            exit("Couldn't load the constraint!")

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

    def configure_optimizer(config):
        optimizer = config['optimizer'][0]
        optimizer.learning_rate = config['learning_rate'][0]
        if config['mixed_precision'][0]:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
        else:
            optimizer.get_scaled_loss = lambda x: x
            optimizer.get_unscaled_gradients = lambda x: x
        return optimizer

    for key, config_original in configurations.items():
        tf.keras.backend.clear_session()

        # Standardize Configuration
        config = copy.deepcopy(config_original)
        config = load_hparams_to_config(hparams, config)
        utils.check_config_combos(config)
        config = utils.format_config_combos(config)
        print(config)
        
        # Get data, network, optimizer, and generate model
        train_data, val_data, transformers = get_preprocessed_data(config)
        dataset, val_dataset = configure_dataset(train_data, val_data, config)
        optimizer = configure_optimizer(config)
        network = load_network(config)
        model = CustomModel(config, network)
        model.compile(optimizer=optimizer, loss="mse")#, run_eagerly=True)#, metrics=["mae"])

        # Train network 
        callback = CustomCallback()
        schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
        lr_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5000, factor=0.5, min_delta=0.00001, verbose=1, min_lr=1E-5)

        history = model.fit(dataset, 
                            epochs=config['epochs'][0], 
                            verbose=0,
                            validation_data=val_dataset,
                            callbacks=[callback, schedule])#lr_on_plateau])# schedule])#, hyper_params])# tensorboard, hyper_params])#,
                                        #early_stop])
        history.history['time_delta'] = callback.time_delta
        model.history = history

        # TODO: Save extra parameters like optimizer.learning_rate
        # Save network and config information
        model.config['time_delta'] = [callback.time_delta]
        model.config['x_transformer'][0] = transformers['x']
        model.config['u_transformer'][0] = transformers['u']
        model.config['a_transformer'][0] = transformers['a']


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
