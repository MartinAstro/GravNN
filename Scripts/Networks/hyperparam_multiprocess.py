#import tensorflow_model_optimization as tfmot
from tensorboard.plugins.hparams import api as hp
import multiprocessing as mp
import pandas as pd
import itertools
from GravNN.Trajectories.ExponentialDist import ExponentialDist

def main():

    # Take trained network and see if a larger learning rate would have let it continue learning.
    df_file = 'Data/Dataframes/exponential_invert_dist_v10.data'
    directory = 'logs/exponential_invert_dist_v10/'

    df_file = 'Data/Dataframes/hyperparameter_moon_pinn_40_v10.data'
    directory = 'logs/hyperparameter_moon_pinn_40_v10/'

    threads = 6
    threads = 1

    hparams = {
        'N_dist' : [5000000],#[1200000],
        'N_train' :[4900000],#[1000000], 
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
        'scale_by' : ['a'],
        'mixed_precision' : [False],
        'num_units' : [40],
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
    import sys
    import numpy as np
    from GravNN.Networks import utils, utils_tf
    from GravNN.Networks.Callbacks import CustomCallback
    from GravNN.Networks.Data import get_preprocessed_data, configure_dataset
    from GravNN.Networks.Model import CustomModel
    from GravNN.Networks.Networks import load_network
    from GravNN.Networks.utils_tf import load_hparams_to_config

    # Get global keywords
    for key, value in hparams.items():
        name = key
        if name == 'decay_rate' : decay_rate = value 
        if name == 'learning_rate': initial_learning_rate = value 
        if name == 'decay_rate_epoch': decay_rate_epochs = value
        if name == 'decay_epoch_0' : decay_epoch_0 = value 
        if name == 'network_shape' : network_shape = value 
        if name == 'mixed_precision' : mixed_precision_flag = value
        if name == 'PINN_constraint_fcn' : PINN_constraint_fcn_val = value
        if name == 'planet' : planet_val = value

    configurations = utils_tf.get_default_config(PINN_constraint_fcn_val, planet_val)

    if sys.platform == 'win32':
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    if mixed_precision_flag:
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print('Compute dtype: %s' % policy.compute_dtype)
        print('Variable dtype: %s' % policy.variable_dtype)

    np.random.seed(1234)
    tf.random.set_seed(0)

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

        # Save network and config information
        model.config['time_delta'] = [callback.time_delta]
        model.config['x_transformer'][0] = transformers['x']
        model.config['u_transformer'][0] = transformers['u']
        model.config['a_transformer'][0] = transformers['a']

        # Appends the model config to a perscribed df
        model.save(df_file=None)
        return model.config


if __name__ == '__main__':
    main()
