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
    # df_file = "Data/Dataframes/fourier_tests_updated.data" 
    config = get_default_eros_config()
    config.update(PINN_III())
    config.update(ReduceLrOnPlateauConfig())

    hparams = {
        "N_dist" : [50000],
        "N_train" : [4500],
        "N_val" : [500],
        "num_units" : [20],
        
        "loss_fcns" : [['percent']],#, 'rms']],
        "jit_compile" : [True],
        "lr_anneal" : [False],
        # "eager" : [True],
        "learning_rate" : [0.0001],
        "dropout" : [0.0],
        "batch_size" : [2**16],
        "epochs" : [10000],
       
        # "network_arch" : ["traditional"],
        # "preprocessing" : [["pines", "r_inv", "fourier_2n"]],
        # "preprocessing" : [["pines", "r_inv", "fourier"]],
        "preprocessing" : [["pines", "r_inv"]],
        "PINN_constraint_fcn" : ['pinn_a'],
        # "init_file" : [2460020.362463469],
        # "fourier_features" : [3],
        # "fourier_sigma" : [2],
        # "freq_decay": [True],
        # "trainable_tanh" : [True],
        # "tanh_r" : [3],
        "tanh_k" : [1E3],
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
    from GravNN.Networks.Saver import ModelSaver
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
    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    print(f"Model ID: [{model.config['id']}]")
    return model.config


if __name__ == "__main__":
    main()
