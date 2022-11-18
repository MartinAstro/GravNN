import multiprocessing as mp
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
from Hyperparam_inits import hyperparams_eros
import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

def main():

    threads = 1

    df_file = "Data/Dataframes/eros_point_mass_v5_3.data" 
    df_file = "Data/Dataframes/test.data" 
    config = get_default_eros_config()

    from GravNN.GravityModels.PointMass import get_pm_data
    from GravNN.GravityModels.Polyhedral import get_poly_data
    from GravNN.CelestialBodies.Asteroids import Eros

    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "grav_file": [Eros().obj_8k],
        "N_dist": [100000],
        "N_train": [9850],
        "N_val": [1500],
        "radius_max" : [Eros().radius*10],
    
        "learning_rate": [0.002],
        "num_units": [20],
        "PINN_constraint_fcn" : ['pinn_a'],
        "patience" : [50],
        'override': [False],
        'ref_radius': [Eros().radius],
        "batch_size" : [2**13],
        "epochs" : [1000],
        "remove_point_mass" : [False],
        "gravity_data_fcn" : [get_pm_data],
        "jit_compile" : [False],
        "dtype" : ['float64'],
        "scale_by" : ['non_dim_v2'],
        "eager" : [False],
        "ref_radius" : [Eros().radius],
        "ref_radius_min" : [Eros().radius_min],
    }

    args = configure_run_args(config, hparams)
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
    print(config)

    # Get data, network, optimizer, and generate model
    data = DataSet(config)
    model = PINNGravityModel(config)
    history = model.train(data)

    model.save(df_file=None, history=history, transformers=data.transformers)
    return model.config


if __name__ == "__main__":
    main()
