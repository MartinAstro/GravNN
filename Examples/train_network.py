import multiprocessing as mp
from GravNN.Networks.script_utils import save_training
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

def main():

    # dataframe where the network configuration info will be saved
    df_file = "Data/Dataframes/example_training.data" 

    # a default set of hyperparameters / configuration details for PINN
    config = get_default_eros_config()

    # hyperparameters which overwrite defaults
    hparams = PINN_III()
    hparams.update(ReduceLrOnPlateauConfig())
    hparams.update({
        "grav_file" : [Eros().obj_8k],
        "N_dist": [5000],
        "N_train": [4500],
        "N_val": [500],
        "batch_size" : [4096],
        "PINN_constraint_fcn": ["pinn_a"],

        # 'trainable_tanh' : [True],
        # 'tanh_k' : [1.0],
        # 'tanh_r' : [1.0],
    })


    threads = 1
    args = configure_run_args(config, hparams)
    with mp.Pool(threads) as pool:
        results = pool.starmap_async(run, args)
        configs = results.get()
    save_training(df_file, configs)


def run(config):
    # Tensorflow dependent functions must be defined inside of 
    # run function for thread-safe behavior.
    from GravNN.Networks.Data import DataSet
    from GravNN.Networks.Model import PINNGravityModel
    from GravNN.Networks.Saver import ModelSaver
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
    saver = ModelSaver(model, history)
    saver.save(df_file=None)

    # Appends the model config to a perscribed df
    return model.config


if __name__ == "__main__":
    main()
