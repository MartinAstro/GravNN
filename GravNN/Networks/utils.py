import os
import zipfile
import tempfile
import itertools
import pandas as pd
from GravNN.Trajectories import ExponentialDist, GaussianDist

def configure_tensorflow():
    set_tf_env_flags()
    tf = set_tf_expand_memory()
    return tf

def set_tf_env_flags():
    import os
    os.environ["PATH"] += os.pathsep + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"
    os.environ["TF_GPU_THREAD_MODE"] ='gpu_private'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

def set_tf_expand_memory():
    import sys
    import tensorflow as tf
    if sys.platform == 'win32':
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    return tf 

def set_mixed_precision():
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    return mixed_precision


def _get_optimizer(name):
    import tensorflow as tf

    # This maintains backwards compatibility for when
    # the entire optimizer object was saved
    if 'adam' in name.lower():
        name = 'adam'
    elif 'rms' in name.lower():
        name = 'rmsprop'
    elif 'sgd' in name.lower():
        name = 'sgd'
    else:
        pass

    return {
        "sgd": tf.keras.optimizers.SGD(),
        "adagrad": tf.keras.optimizers.Adagrad(),
        "adadelta": tf.keras.optimizers.Adadelta(),
        "rmsprop": tf.keras.optimizers.RMSprop(),
        "adam": tf.keras.optimizers.Adam(),
    }[name.lower()]


def _get_annealing_fcn(value):
    from GravNN.Networks.Annealing import update_constant, hold_constant
    if value:
        return update_constant
    else:
        return hold_constant

def _get_acceleration_nondim_constants(value, config):
    import tensorflow as tf
    # Backwards compatibility (if the value is a function -- take the name of the function then select corresponding values)
    try:
        value = value.__name__
    except:
        pass
    
    a_bar_s = config['a_bar_transformer'][0].scale_
    a_bar_0 = config['a_bar_transformer'][0].min_

    a0 = a_bar_0
    a_s = a_bar_s

    u_xx_s = config['u_transformer'][0].scale_
    u_xx_0 = config['u_transformer'][0].min_

    x_s = config['x_transformer'][0].scale_
    x_0 = config['x_transformer'][0].min_

    l_s = x_s**2
    c_s = x_s**2

    l_s = 1.0
    c_s = 1.0



    # scale tensor + translate tensor
    return {
        "no_pinn": (tf.constant([a_s, a_s, a_s], dtype=tf.float32), tf.constant([a0, a0, a0], dtype=tf.float32)), # scaling ignored
        "pinn_a": (tf.constant([a_s, a_s, a_s], dtype=tf.float32), tf.constant([a0, a0, a0], dtype=tf.float32)), # scaling ignored
        "pinn_p": (tf.constant([1.0], dtype=tf.float32), tf.constant([0.0], dtype=tf.float32)),
        
        "pinn_pl": (tf.constant([1.0, l_s], dtype=tf.float32), tf.constant([0.0, 0.0], dtype=tf.float32)),
        "pinn_plc": (tf.constant([1.0, l_s, c_s, c_s, c_s], dtype=tf.float32), tf.constant([0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32)),

        "pinn_ap": (tf.constant([1.0, a_s, a_s, a_s], dtype=tf.float32), tf.constant([0.0, a0, a0, a0], dtype=tf.float32)),
        "pinn_al": (tf.constant([a_s, a_s, a_s, l_s], dtype=tf.float32), tf.constant([a0, a0, a0, 0.0], dtype=tf.float32)),
        "pinn_alc": (tf.constant([a_s, a_s, a_s, l_s, c_s, c_s, c_s], dtype=tf.float32), tf.constant([a0, a0, a0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32)),
        "pinn_apl": (tf.constant([1.0, a_s, a_s, a_s, l_s], dtype=tf.float32), tf.constant([0.0, a0, a0, a0, 0.0], dtype=tf.float32)),
        "pinn_aplc": (tf.constant([1.0, a_s, a_s, a_s, l_s, c_s, c_s, c_s], dtype=tf.float32), tf.constant([0.0, a0, a0, a0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32)),

        "pinn_a_sph": (tf.constant([a_s, a_s, a_s], dtype=tf.float32), tf.constant([a0, a0, a0], dtype=tf.float32)) # scaling ignored
    }[value.lower()]


def _get_PI_constraint(value):
    from GravNN.Networks.Constraints import no_pinn, pinn_A, pinn_P, pinn_PLC, pinn_AP, \
        pinn_AL, pinn_ALC, pinn_APL, pinn_APLC, pinn_A_sph
    from GravNN.Networks.Annealing import no_pinn_anneal, pinn_A_anneal, pinn_P_anneal, pinn_AP_anneal, pinn_AL_anneal, pinn_ALC_anneal, pinn_APL_anneal, pinn_APLC_anneal, pinn_PLC_anneal

    # Backwards compatibility (if the value is a function -- take the name of the function then select corresponding values)
    try:
        value = value.__name__
    except:
        pass

    return {
        "no_pinn": [no_pinn, no_pinn_anneal, [-1.0]], # scaling ignored
        "pinn_a": [pinn_A, pinn_A_anneal, [-1.0]], # scaling ignored
        "pinn_p": [pinn_P, pinn_P_anneal, [-1.0]],
        
        "pinn_pl": [pinn_P, pinn_P_anneal, [-1.0, 1.0]],
        "pinn_plc": [pinn_PLC, pinn_PLC_anneal, [-1.0, 1.0, 1.0]],

        "pinn_ap": [pinn_AP, pinn_AP_anneal, [-1.0, 1.0]],
        "pinn_al": [pinn_AL, pinn_AL_anneal, [-1.0, 1.0]],
        "pinn_alc": [pinn_ALC, pinn_ALC_anneal, [-1.0, 1.0, 1.0]],
        "pinn_apl": [pinn_APL, pinn_APL_anneal, [-1.0, 1.0, 1.0]],
        "pinn_aplc": [pinn_APLC, pinn_APLC_anneal, [-1.0, 1.0, 1.0, 1.0]],

        "pinn_a_sph": [pinn_A_sph, pinn_A_anneal, [-1.0]] # scaling ignored

    }[value.lower()]

def _get_network_fcn(name):
    from GravNN.Networks.Networks import TraditionalNet, ResNet, SphericalTraditionalNet, SphericalPinesTraditionalNet, CylindricalTraditionalNet, SphericalPinesTransformerNet
    return {
        "traditional": TraditionalNet,
        "resnet": ResNet,
        "sph_traditional": SphericalTraditionalNet,
        "sph_pines_traditional": SphericalPinesTraditionalNet,
        "sph_pines_transformer": SphericalPinesTransformerNet,
        "cyl_traditional": CylindricalTraditionalNet
    }[name.lower()]

def _get_tf_dtype(name):
    import tensorflow as tf
    return {
        'float16' : tf.float16,
        'float32' : tf.float32,
        'float64' : tf.float64
    }[name.lower()]

def load_hparams_to_config(hparams, config):

    for key, value in hparams.items():
        config[key] = [value]

    config['PINN_constraint_fcn'] = _get_PI_constraint(config['PINN_constraint_fcn'][0])    
    config['network_type'] = [_get_network_fcn(config['network_type'][0])]
    config['dtype'] = [_get_tf_dtype(config['dtype'][0])]
    
    if 'num_units' in config:
        for i in range(1, len(config['layers'][0])-1):
            config['layers'][0][i] = config['num_units'][0]
            
    return config


def configure_optimizer(config, mixed_precision):
    optimizer = _get_optimizer(config['optimizer'][0])
    optimizer.learning_rate = config['learning_rate'][0]
    if config['mixed_precision'][0]:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
    else:
        optimizer.get_scaled_loss = lambda x: x
        optimizer.get_unscaled_gradients = lambda x: x
    return optimizer



def configure_run_args(config, hparams):
    keys, values = zip(*hparams.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    args = []
    session_num = 0
    for hparam_inst in permutations_dicts:
        print('--- Starting trial: %d' % session_num)
        print({key: value for key, value in hparam_inst.items()})
        args.append((config, hparam_inst))
        session_num += 1
    return args


def get_gzipped_model_size(model):
    # Returns size of gzipped model, in bytes.
    _, keras_file = tempfile.mkstemp('.h5')
    model.network.save(keras_file, include_optimizer=False)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)

    return os.path.getsize(zipped_file)


def check_config_combos(config):
    from GravNN.Networks.Constraints import no_pinn
    if config['PINN_constraint_fcn'][0] != no_pinn:
        if config['layers'][0][-1] != 1:
            print("WARNING: The final layer for a PINN must have one output (the potential, U) -- changing automatically")
            config['layers'][0][-1] = 1 
    else:
        if config['layers'][0][-1] != 3:
            config['layers'][0][-1] = 3
            print("WARNING: The final layer for a traditional network must have three outputs (the acceleration vector, a) -- changing automatically")
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
    directory = os.path.abspath('.') +"/Data/Dataframes/"
    os.makedirs(directory, exist_ok=True)
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

def update_df_row(model_id, df_file, entries, save=True):
    """Update a row in the dataframe

    Args:
        model_id (float): Timetag for model within dataframe
        df_file (any): Either the path used to load the df (slow) or the df itself (fast)
        entries (series): The series to update in the df
        save (bool, optional): Save the dataframe immediately after updating (slow). Defaults to True.

    Returns:
        DataFrame: The updated dataframe
    """
    if type(df_file) == str:
        original_df = pd.read_pickle(df_file)
    else:
        original_df = df_file
    timestamp = pd.to_datetime(model_id, unit = 'D', origin = 'julian').round('s').ctime()
    entries.update({"timetag" : [timestamp]})
    dictionary = dict(sorted(entries.items(), key = lambda kv: kv[0]))
    df = pd.DataFrame.from_dict(dictionary).set_index('timetag')
    original_df = original_df.combine_first(df)
    original_df.update(df)#, sort=True) # join, merge_ordered also viable
    if save:
        original_df.to_pickle(df_file)
    return original_df

