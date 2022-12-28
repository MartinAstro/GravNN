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

    threads = 6

    df_file = "Data/Dataframes/multiFF_hparams.data" 
    df_file = "Data/Dataframes/multiFF_followup.data" 

    config = get_default_earth_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
    
        "radius_min" : [Earth().radius*1],
        "radius_max" : [Earth().radius*1.06],
        "ref_radius_min" : [Earth().radius*1],
        "ref_radius_max" : [Earth().radius*1.06],
        "feature_min" : [1.0],
        "feature_max" : [1.0 + (Earth().radius*1.06)/Earth().radius],
        "N_dist": [5000],
        # "N_train": [2000],
        "N_train": [1000],
        "N_val" : [100],
        "batch_size" : [2**20],
        # "radius_max" : [Earth().radius*5],
        # "N_dist": [1100000],
        # "N_train": [1000000],
        # "batch_size" : [2**20],

        "radius_max" : [Earth().radius + 420000],
        "N_dist": [500000],
        "N_train": [20000],
        "N_val": [5000], 
        "batch_size" : [2**20],

        "learning_rate": [0.001],
        "PINN_constraint_fcn" : ['pinn_a'],

        "epochs" : [10000],

        "ref_radius_max" : [Earth().radius+420000.0],
        "ref_radius_min" : [Earth().radius],

        "deg_removed" : [2],  #-1
        "remove_point_mass" : [False],
        "jit_compile" : [True],
        "eager" : [False],
        "loss_sph" : [False],

        "num_units": [10],
        "fourier_features" : [20],
        "fourier_sigma" : [
            [1.0, 2.0],
            ],

        "network_type": ["custom"],        

        "loss_fcns" : [
            # ['rms', 'percent'],
            # ['rms'],
            ['percent'],
        ],

        # TBD
        "lr_anneal": ['hold'],
        "beta": [0.001],

        # Investigate Later
        "uniform_volume" : [False],
        "dropout" : [0.00],


    }
    args = configure_run_args(config, hparams)
    # run(*args[0])
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)


def run(config):    
    from GravNN.Networks.Data import DataSet
    from GravNN.Networks.Model import PINNGravityModel
    from GravNN.Networks.utils import configure_tensorflow
    from GravNN.Networks.utils import populate_config_objects

    configure_tensorflow(config)

    # Standardize Configuration
    config = populate_config_objects(config)
    pprint(config)

    from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
    grav_model = SphericalHarmonics(Earth().EGM2008,2)
    config['cBar'] = [grav_model.C_lm]
    config['sBar'] = [grav_model.S_lm]

    # Get data, network, optimizer, and generate model
    data = DataSet(config)
    model = PINNGravityModel(config)
    history = model.train(data)

    model.save(df_file=None, history=history, transformers=data.transformers)
    return model.config


if __name__ == "__main__":
    main()
