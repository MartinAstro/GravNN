import multiprocessing as mp
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
import os
from pprint import pprint
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

def main():

    threads = 4

    df_file = "Data/Dataframes/example.data" 
    df_file = "Data/Dataframes/fourier_tests_updated.data" 
    config = get_default_earth_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
<<<<<<< HEAD
        "N_dist" : [50000],
        # "N_train" : [2],
        # "N_val" : [2],
        "N_train" : [450],
        "N_val" : [50],
        # "learning_rate" : [1E-9], # some discussion that learning rate must be small for dynamics 
        "num_units" : [10],
=======
        "N_dist" : [1000000],
        # "radius_max" : [Earth().radius + 1.0],
        # "N_train" : [2],
        # "N_val" : [2],
        "N_train" : [45000, 450000],
        "N_val" : [500],
        # "learning_rate" : [1E-9], # some discussion that learning rate must be small for dynamics 
        # "num_units" : [128],
        "num_units" : [20, 80],
>>>>>>> 8672f90 (Add main to scripts to call after training loop)
        # "layers" : [[3, 3, 2, 1]],
        "loss_fcns" : [['rms']],
        # "jit_compile" : [False],
        # "eager" : [True],
<<<<<<< HEAD
        "PINN_constraint_fcn" : ['pinn_alc'],
        "epochs" : [5000]
=======
        "learning_rate" : [0.0001],
        "dropout" : [0.0, 0.01],
        "batch_size" : [2**15],
        # "activation": ['tanh'],
        # "PINN_constraint_fcn" : ['pinn_alc'],
        # "network_arch" : ["traditional"],
        "preprocessing" : [["pines", "r_inv", "fourier_2n"]],
        # "preprocessing" : [["pines", "r_inv", "fourier"]],
        # "preprocessing" : [["pines", "r_inv"]],#, "fourier"]],
        "PINN_constraint_fcn" : ['pinn_a'],
        # "batch_size" : [4096],
        # "trainable_tanh" : [True],
        "fourier_features" : [2,5,10],
        "fourier_sigma" : [2],
        "freq_decay": [True,False],
        "epochs" : [5000],
        "tanh_k" : [1, 1E3],
        # "tanh_r" : [3],

        # "dtype" : ['float64']
>>>>>>> 8672f90 (Add main to scripts to call after training loop)
    }
    args = configure_run_args(config, hparams)
    # run(*args[0])
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)

    # from Scripts.Figures.Earth.nn_brillouin_map import main as main_maps
    # main_maps()
    # from GravNN.Visualization.HistoryVisualizer import main as main_history
    # main_history()


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
