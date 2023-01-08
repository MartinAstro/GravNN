import multiprocessing as mp
from GravNN.Analysis.PlanesExperiment import PlanesExperiment
from GravNN.Analysis.ExtrapolationExperiment import ExtrapolationExperiment
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
from GravNN.Visualization.ExtrapolationVisualizer import ExtrapolationVisualizer
from GravNN.Visualization.PlanesVisualizer import PlanesVisualizer
import os
from pprint import pprint
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

def main():

    threads = 8

    df_file = "Data/Dataframes/earth_trainable_FF.data" 

    # config = get_default_eros_config()
    config = get_default_earth_config()
    # config = get_default_moon_config()

    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "PINN_constraint_fcn" : ['pinn_a'],

        "N_dist": [50000],
        "N_train": [10000],
        "N_val": [500],
        "epochs" : [5000],

        "remove_point_mass" : [False],
        "jit_compile" : [True],
        "eager" : [False],
        # "jit_compile" : [False],
        # "eager" : [True],

        # "dtype" : ['float64'],
        # "dropout" 
        # "activation"
        # "loss_fcns"

        # # Features to Investigate

        "learning_rate": [0.01],
        "deg_removed" : [-1],

        "network_arch" : ["transformer"],
        "activation": ['tanh'],
        "batch_size" : [2**10],

        # "network_type": ["siren"],
        # "learning_rate": [0.0005],

        "network_type": ["custom"],
        # "network_type": ["multi"],
        # "fourier_features" : [10],
        # "fourier_sigma" : [[1]],

        # "preprocessing" : [[]],
        # "scale_nn_potential" : [False],
        # "loss_sph" : [True],
        # "deg_removed" : [-1],
                        
        # Investigate Later
        "uniform_volume" : [False],
        # "preprocessing"

        # To be completed
        "lr_anneal": ['hold'],
        "beta": [0.001],
    }
    args = configure_run_args(config, hparams)
    # run(*args[0])
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)


def run(config):    

    def experiment(config, model, radius_bounds, max_percent):
        planes_exp = PlanesExperiment(model, config, radius_bounds, 30)
        planes_exp.run()
        vis = PlanesVisualizer(planes_exp, save_directory=os.path.abspath(".")+"/Plots/Eros/", halt_formatting=True)
        vis.plot(percent_max=max_percent)

    def extrapolation_experiment(config, model):
        extrap_exp = ExtrapolationExperiment(model, config, 500)
        extrap_exp.run()
        vis = ExtrapolationVisualizer(extrap_exp)
        vis.plot_interpolation_percent_error()
        vis.plot_extrapolation_percent_error()

    from GravNN.Networks.Data import DataSet
    from GravNN.Networks.Model import PINNGravityModel
    from GravNN.Networks.utils import configure_tensorflow
    from GravNN.Networks.utils import populate_config_objects

    import time
    import numpy as np
    import os

    configure_tensorflow(config)

    # Standardize Configuration
    config = populate_config_objects(config)
    pprint(config)

    # import tensorflow as tf
    # print(tf.config.list_physical_devices('GPU'))
    # tf.debugging.set_log_device_placement(True)
    # tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

    # from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
    # grav_model = SphericalHarmonics(Earth().EGM2008,2)
    # config['cBar'] = [grav_model.C_lm]
    # config['sBar'] = [grav_model.S_lm]

    # Get data, network, optimizer, and generate model
    np.random.seed(1234)

    data = DataSet(config)
    model = PINNGravityModel(config)
    # np.random.seed(int(os.getpid() % 13))
    time.sleep(np.random.uniform(0, 5))#))

    # if transfer
    # for layer in model.network.layers[:4]:
    #     layer.trainable = False

    history = model.train(data)

    # radius = config['planet'][0].radius
    # radius_bounds = [-1*radius, 1*radius]
    # experiment(config, model, radius_bounds, max_percent=10)
    # radius_bounds = [-10*radius, 10*radius]
    # experiment(config, model, radius_bounds, max_percent=10)
    # radius_bounds = [-100*radius, 100*radius]
    # experiment(config, model, radius_bounds, max_percent=10)
    # extrapolation_experiment(config, model)


    # fine tuning
    if 'fine_tuning_epochs' in config:
        config['epochs'] = [config['fine_tuning_epochs'][0]]
        model.set_training(False) # remove dropout 
        history = model.train(data, initialize_optimizer=False)

    # radius = config['planet'][0].radius
    # radius_bounds = [-1*radius, 1*radius]
    # experiment(config, model, radius_bounds, max_percent=10)
    # radius_bounds = [-10*radius, 10*radius]
    # experiment(config, model, radius_bounds, max_percent=10)
    # radius_bounds = [-100*radius, 100*radius]
    # experiment(config, model, radius_bounds, max_percent=10)
    # extrapolation_experiment(config, model)

    # import matplotlib.pyplot as plt
    # # plt.figure()
    # # plt.plot(history.history['loss'])
    # # plt.plot(history.history['val_loss'])
    # # plt.figure()
    # # plt.plot(history.history['percent_mean'])
    # # plt.plot(history.history['val_percent_mean'])
    # plt.show()
    # plt.close()

    # vis.save(plt.gcf(), "Eros_Planes.pdf")


    model.save(df_file=None, history=history, transformers=data.transformers)
    return model.config


if __name__ == "__main__":
    main()
