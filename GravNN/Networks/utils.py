import os
import zipfile
import tempfile
import pandas as pd
import tensorflow as tf
#import tensorflow_model_optimization as tfmot
from GravNN.Trajectories.ExponentialDist import ExponentialDist
from GravNN.Trajectories.GaussianDist import GaussianDist




def get_gzipped_model_size(model):
    # Returns size of gzipped model, in bytes.
    _, keras_file = tempfile.mkstemp('.h5')
    model.network.save(keras_file, include_optimizer=False)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)

    return os.path.getsize(zipped_file)

def count_nonzero_params(model):
    params = 0
    for v in model.trainable_variables:
        params += tf.math.count_nonzero(v)
    return params.numpy()

def check_config_combos(config):
    if config['PINN_flag'][0] != "none":
        assert config['layers'][0][-1] == 1, "If PINN, the final layer must have one output (the potential, U)"
    else:
        assert config['layers'][0][-1] == 3, "If not PINN, the final layer must have three outputs (the acceleration vector, a)"
    if config['network_type'][0].__class__.__name__ == "InceptionNet":
        assert len(config['layers'][0][1]) != 0, "Inception network requires layers with multiple sizes, i.e. [[3, [3,7,11], [3,7,11], 1]]"


def format_config_combos(config):
    # Ensure distributions don't have irrelevant parameters defined
    if config['distribution'][0] == GaussianDist:
        config['invert'] = [None]
        config['scale_parameter'] = [None]

    if config['distribution'][0] == ExponentialDist:
        config['mu'] = [None]
        config['sigma'] = [None]
    
    return config

def save_df_row(dictionary, df_file):
    dictionary = dict(sorted(dictionary.items(), key = lambda kv: kv[0]))
    df = pd.DataFrame().from_dict(dictionary).set_index('timetag')
    try: 
        df_all = pd.read_pickle(df_file)
        df_all = df_all.append(df)
        df_all.to_pickle(df_file)
    except: 
        df.to_pickle(df_file)

def get_df_row(model_id, df_file):
    original_df = pd.read_pickle(df_file)
    config = original_df[model_id == original_df['id']].to_dict()
    for key, value in config.items():
        config[key] = list(value.values())
    return config

def update_df_row(model_id, df_file, entries):
    original_df = pd.read_pickle(df_file)
    timestamp = pd.to_datetime(model_id, unit = 'D', origin = 'julian').round('s').ctime()
    entries.update({"timetag" : [timestamp]})
    dictionary = dict(sorted(entries.items(), key = lambda kv: kv[0]))
    df = pd.DataFrame.from_dict(dictionary).set_index('timetag')
    original_df = original_df.combine_first(df)
    original_df.update(df)#, sort=True) # join, merge_ordered also viable
    original_df.to_pickle(df_file)
    return 

def save_df_column(entries, index_name, column_name, df_file):
    # ! This doesn't add a column -- fix this
    dictionary = dict(sorted(entries.items(), key = lambda kv: kv[0]))
    df = pd.DataFrame().from_dict(entries, orient='columns').set_index('alt')
    try: 
        df_all = pd.read_pickle(df_file)
        df_all = df_all.append(df)
        df_all.to_pickle(df_file)
    except: 
        df.to_pickle(df_file)


    # original_df = pd.read_pickle(df_file)
    # df = pd.DataFrame.from_dict(config).set_index('model_id')
    # original_df = original_df.merge(df, sort=True) # join, merge_ordered also viable
    # original_df.to_pickle(df_file)


# def check_divergent(x_unscaled, a_unscaled):
    # non_divergent_idx = (x_unscaled[:,2] != 0 or x_unscaled[:,2] != np.deg2rad(180.0))
    # x_unscaled = x_unscaled[non_divergent_idx] 
    # a_unscaled = a_unscaled[non_divergent_idx]