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

    df_file = "Data/Dataframes/multiFF_hparams.data" 
    df_file = "Data/Dataframes/multiFF_followup.data" 
    df_file = "Data/Dataframes/best_10_followup.data" 
    df_file = "Data/Dataframes/earth_high_alt.data" 
    df_file = "Data/Dataframes/earth_high_alt3.data" 
    df_file = "Data/Dataframes/earth_high_alt4.data" 
    df_file = "Data/Dataframes/new_hparam_search.data" 
    df_file = "Data/Dataframes/hparams_ll.data" 
    df_file = "Data/Dataframes/hparams_ll2.data" 
    df_file = "Data/Dataframes/test_uniform.data" 
    df_file = "Data/Dataframes/sigma_search.data" 
    df_file = "Data/Dataframes/alc.data" 
    df_file = "Data/Dataframes/fourier_search_2.data" 
    df_file = "Data/Dataframes/high_altitude_behavior.data" 
    df_file = "Data/Dataframes/eros_PINN_III_temp.data" 
    df_file = "Data/Dataframes/eros_PINN_III.data" 
    df_file = "Data/Dataframes/eros_PINN_III_hparams.data" 
    df_file = "Data/Dataframes/earth_percent_error_test.data" 
    df_file = "Data/Dataframes/eros_PINN_II_hparams.data" 

    config = get_default_eros_config()
    # config = get_default_earth_config()
    # config = get_default_earth_config()
    config.update(PINN_II())
    config.update(ReduceLrOnPlateauConfig())

    # import numpy as np
    # import itertools
    # FF1 = [2**(i) for i in range(-3, 3)]
    # FF2 = [2**(i) for i in range(-3, 3)]
    # sigmas = np.array(list(itertools.product(FF1, FF2)))


    hparams = {


        "batch_size" : [2**20],
        "PINN_constraint_fcn" : ['pinn_a'],

        # "N_dist": [50000],
        # "N_train": [2000],
        # "N_val": [500],
        # "epochs" : [5000],


        "N_dist": [50000],
        "N_train": [2**11, 2**12, 2**13, 2**14, 2**15],
        "epochs" : [2**10, 2**11, 2**12, 2**13, 2**14],
        "num_units": [10, 20, 40, 80],
        "N_val": [5000],


        "learning_rate": [0.001],
        # "num_units" : [20],

        "remove_point_mass" : [False],
        "jit_compile" : [True],
        "eager" : [False],
        # "jit_compile" : [False],
        # "eager" : [True],

        # "loss_fcns" : [['percent']],
        # "dropout" : [0.1],


        # "dropout" 
        # "activation"
        # "loss_fcns"

        # Features to Investigate
        "network_type": ["custom"],
        "loss_sph" : [False],
        
        
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
    data = DataSet(config)
    model = PINNGravityModel(config)
    np.random.seed(int(os.getpid() % 13))
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
