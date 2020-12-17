import os
import zipfile
import tempfile
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot

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
    if config['PINN_flag'][0] == True:
        assert config['layers'][0][-1] == 1, "If PINN, the final layer must have one output (the potential, U)"
    else:
        assert config['layers'][0][-1] == 3, "If not PINN, the final layer must have three outputs (the acceleration vector, a)"
    if config['network_type'][0].__class__.__name__ == "InceptionNet":
        assert len(config['layers'][0][1]) != 0, "Inception network requires layers with multiple sizes, i.e. [[3, [3,7,11], [3,7,11], 1]]"
    

def save_dataframe_row(dictionary, df_file):
    dictionary = dict(sorted(dictionary.items(), key = lambda kv: kv[0]))
    df = pd.DataFrame().from_dict(dictionary).set_index('timetag')
    timestamp = str(pd.Timestamp(dictionary['timetag'][0]).to_julian_date()) 
    directory = os.path.abspath('.') +"/Plots/"+ timestamp + "/"
    try: 
        df_all = pd.read_pickle(df_file)
        df_all = df_all.append(df)
        df_all.to_pickle(df_file)
    except: 
        df.to_pickle(df_file)


def load_config(file_name, timetag):
    nn_df = pd.read_pickle(file_name)
    config = nn_df[nn_df['timetag'] == timetag].to_dict()
    config['init_file'] = timetag
    for key, value in config.items():
        config[key] = [value]
    return config
