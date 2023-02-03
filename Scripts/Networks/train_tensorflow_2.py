import multiprocessing as mp
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
import os
from pprint import pprint
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

def main():

    threads = 8

    df_file = "Data/Dataframes/example.data" 
    config = get_default_earth_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist" : [50000],
        "N_train" : [45000],
        "N_val" : [500],
        "radius_min": [Earth().radius],
        "radius_max": [Earth().radius + 1.0],
        "epochs"  :[5000],

        # "loss_fcns" : [['rms']],
        # "jit_compile" : [False],
        "epochs" : [5000],
        # "eager" : [True], # saving an eager net causes issues
        "PINN_constraint_fcn" : ['pinn_a'],
        "network_arch" : ["convolutional"],
        # "network_arch" : ["traditional"],
        # "layers" : [[3, 80, 40, 20, 10, 1]]
        # "layers" : [[3, 80, 20, 20, 80, 1]]
        # "layers" : [[3, 20, 20, 20, 20, 1]]
        # "num_units" : [256],
        "num_units" : [128],
        "preprocessing" : [['pines', 'r_inv', 'fourier']],
        "fourier_sigma" : [[1.0]],
        "fourier_features" :[48],
        "batch_size" : [8192]

        # "preprocessing" : [['pines', 'r_inv', 'fourier']],
        # "fourier_features" : [20],
        # "fourier_sigma" : [[2.0]],
        # "freq_decay" : [False]

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

    # Get data, network, optimizer, and generate model
    data = DataSet(config)
    model = PINNGravityModel(config)
    history = model.train(data)

    model.save_custom(df_file=None, history=history, transformers=data.transformers)
    return model.config


if __name__ == "__main__":
    main()
