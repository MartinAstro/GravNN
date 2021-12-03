import numpy as np
import pandas as pd
import os
import multiprocessing as mp
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Regression.BLLS import BLLS
from GravNN.Regression.utils import save, format_coefficients
from GravNN.Networks.Data import get_preprocessed_data
from GravNN.Networks.Configs import *
np.random.seed(1234)

def regress_sh_model(config, max_deg, noise, idx):
    # Get randomly shuffled data 
    planet = config['planet'][0]
    dataset, val_dataset, transformers = get_preprocessed_data(config)
    x = dataset[0]
    a = dataset[2]

    file_name = "regress_%d_%.1f_%d.csv" %(max_deg, noise, idx)
    grav_file = os.path.join(os.path.abspath('.') , 'GravNN','Files', 'GravityModels','Regressed', file_name)
    regressor = BLLS(max_deg, planet, remove_deg=2)#
    results = regressor.update(x, a)
    C_lm, S_lm = format_coefficients(results, max_deg, 2)
    save(grav_file, planet, C_lm, S_lm)
    return file_name, max_deg, noise, idx

def generate_args(config, num_models, noise_list, deg_list):
    args = []
    for idx in range(num_models):
        for noise in noise_list:
            for deg in deg_list:
                config['acc_noise'] = [noise]
                config['seed'] = [idx]
                args.append((config, deg, noise, idx))
    return args


def main():
    """Multiprocessed version of generate models. Trains multiple networks 
    simultaneously to speed up regression.
    """
    config = get_default_earth_config()
    config['N_train'] = [9500]
    config['N_val'] = [500]
    config['scale_by'] = ["none"]
    model_deg = 80 
    model_interval = 5
    num_models = 10

    noise_list = [0.0, 0.2]
    deg_list = np.arange(3, model_deg, model_interval, dtype=int)
    model_id_list = np.arange(0, num_models, 1, dtype=int)

    sh_df = pd.DataFrame(index=pd.MultiIndex.from_product([noise_list, deg_list, model_id_list], names=['noise', 'degree', 'id']), columns=['model_identifier'])

    pool = mp.Pool(6)
    args = generate_args(config, num_models, noise_list, deg_list)
    results = pool.starmap_async(regress_sh_model, args)
    tuples = results.get()
    for tuple in tuples:
        sh_model_name, deg, noise, idx = tuple
        sh_df.loc[(noise, deg, idx)] = sh_model_name

    sh_df.to_pickle("Data/Dataframes/regress_sh_2.data")

if __name__ == "__main__":
    main()

