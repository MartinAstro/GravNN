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
        # "N_train" : [2],
        # "N_val" : [2],
        "N_train" : [450],
        "N_val" : [50],
        # "learning_rate" : [1E-9], # some discussion that learning rate must be small for dynamics 
        "num_units" : [10],
        # "layers" : [[3, 3, 2, 1]],
        "loss_fcns" : [['rms']],
        # "jit_compile" : [False],
        # "eager" : [True],
        "network_arch" : ['traditional'],
        "PINN_constraint_fcn" : ['pinn_a'],
    }
    args = configure_run_args(config, hparams)
    # run(*args[0])
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
