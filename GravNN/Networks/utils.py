import os
import zipfile
import tempfile
import itertools
from colorama.ansi import Back
import pandas as pd
from colorama import Fore, init, deinit
from copy import deepcopy
from GravNN.Trajectories import ExponentialDist, GaussianDist


def configure_tensorflow(hparams):
    """Custom tensorflow import that configures proper flags, path settings, 
    seeds, etc.

    Returns:
        module: Tensorflow as tf
    """
    set_tf_env_flags()
    tf = set_tf_expand_memory()
    tf.keras.backend.clear_session()
    tf.random.set_seed(hparams['seed'][0])
    tf.config.run_functions_eagerly(hparams['eager'][0])
    mixed_precision = set_mixed_precision() if hparams['mixed_precision'][0] else None

    return tf, mixed_precision


def set_tf_env_flags():
    """Add CUDA library to path (assumes using v10.1) and configure GPU and XLA."""
    import os

    os.environ["PATH"] += (
        os.pathsep
        + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"
    )
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"


def set_tf_expand_memory():
    """Allow multiple TF processes to run on GPU by allowing memory growth

    Returns:
        module: Tensorflow as tf
    """
    import sys
    import tensorflow as tf

    if sys.platform == "win32":
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    return tf


def set_mixed_precision():
    """Method used to configure mixed precision settings. This allows for faster training times
    for non-physics informed neural networks.

    .. warn:: Do not configure mixed precision when training PINNs. Because gradients of the network
    are embedded within the loss function, the cruder precision can cause convergence issues.

    Returns:
        module: mixed precision module
    """
    from tensorflow.keras.mixed_precision import experimental as mixed_precision

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)
    print("Compute dtype: %s" % policy.compute_dtype)
    print("Variable dtype: %s" % policy.variable_dtype)
    return mixed_precision


def _get_optimizer(name):
    """Helper function to get proper optimizer based on variable stored within
    configuration / hyperparameter dictionary

    Args:
        name (str): name of optimizer

    Returns:
        tf.Optimizer: optimizer object
    """
    import tensorflow as tf

    # This maintains backwards compatibility for when
    # the entire optimizer object was saved
    if "adam" in name.lower():
        name = "adam"
    elif "rms" in name.lower():
        name = "rmsprop"
    elif "sgd" in name.lower():
        name = "sgd"
    else:
        pass

    return {
        "sgd": tf.keras.optimizers.SGD(),
        "adagrad": tf.keras.optimizers.Adagrad(),
        "adadelta": tf.keras.optimizers.Adadelta(),
        "rmsprop": tf.keras.optimizers.RMSprop(),
        "adam": tf.keras.optimizers.Adam(),
    }[name.lower()]


def _get_annealing_fcn(name):
    """Helper function to determine if the annealing learning rates of Wang2020
    are going to be used

    Args:
        name (str): key specifying how lr will be annealed

    Returns:
        function: lr annealing method
    """
    from GravNN.Networks.Annealing import update_constant, hold_constant, custom_constant
    return {
        "anneal": update_constant,
        "hold": hold_constant,
        "custom": custom_constant,
    }[name.lower()]


def _get_acceleration_nondim_constants(value, config):
    """Method responsible how much to scale the values passed for the loss function

    Note that the length of returned list will correspond to the number of loss terms used to train the network.
    e.g. if the network is trained with the potential and acceleration the list will be [u, a1, a2, a3] rather than
    [a1,a2,a3].

    .. todo:: Confirm proper scaling of the L and C values.

    Args:
        value (str): PINN constraint type
        config (dict): dictionary containing hyperparameter and configuration variables

    Returns:
        tf.constant: tensor of values used to scale the loss function into the appropriate [-1,1] range.
    """
    import tensorflow as tf

    # Backwards compatibility (if the value is a function -- take the name of the function then select corresponding values)
    try:
        value = value.__name__
    except:
        pass

    a_bar_s = config["a_bar_transformer"][0].scale_
    a_bar_0 = config["a_bar_transformer"][0].min_

    a0 = a_bar_0
    a_s = a_bar_s

    u_xx_s = config["u_transformer"][0].scale_
    u_xx_0 = config["u_transformer"][0].min_

    x_s = config["x_transformer"][0].scale_
    x_0 = config["x_transformer"][0].min_

    l_s = x_s ** 2
    c_s = x_s ** 2

    l_s = 1.0
    c_s = 1.0

    # scale tensor + translate tensor
    return {
        "no_pinn": (
            tf.constant([a_s, a_s, a_s], dtype=tf.float32),
            tf.constant([a0, a0, a0], dtype=tf.float32),
        ),  # scaling ignored
        "pinn_a": (
            tf.constant([a_s, a_s, a_s], dtype=tf.float32),
            tf.constant([a0, a0, a0], dtype=tf.float32),
        ),  # scaling ignored
        "pinn_p": (
            tf.constant([1.0], dtype=tf.float32),
            tf.constant([0.0], dtype=tf.float32),
        ),
        "pinn_pl": (
            tf.constant([1.0, l_s], dtype=tf.float32),
            tf.constant([0.0, 0.0], dtype=tf.float32),
        ),
        "pinn_plc": (
            tf.constant([1.0, l_s, c_s, c_s, c_s], dtype=tf.float32),
            tf.constant([0.0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32),
        ),
        "pinn_ap": (
            tf.constant([1.0, a_s, a_s, a_s], dtype=tf.float32),
            tf.constant([0.0, a0, a0, a0], dtype=tf.float32),
        ),
        "pinn_al": (
            tf.constant([a_s, a_s, a_s, l_s], dtype=tf.float32),
            tf.constant([a0, a0, a0, 0.0], dtype=tf.float32),
        ),
        "pinn_alc": (
            tf.constant([a_s, a_s, a_s, l_s, c_s, c_s, c_s], dtype=tf.float32),
            tf.constant([a0, a0, a0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32),
        ),
        "pinn_apl": (
            tf.constant([1.0, a_s, a_s, a_s, l_s], dtype=tf.float32),
            tf.constant([0.0, a0, a0, a0, 0.0], dtype=tf.float32),
        ),
        "pinn_aplc": (
            tf.constant([1.0, a_s, a_s, a_s, l_s, c_s, c_s, c_s], dtype=tf.float32),
            tf.constant([0.0, a0, a0, a0, 0.0, 0.0, 0.0, 0.0], dtype=tf.float32),
        ),
    }[value.lower()]


def _get_PI_constraint_all(value):
    """Method responsible for getting all variables / methods used in the physics informed constraint.

    Args:
        value (str): PINN constraint name (i.e. 'pinn_A', 'pinn_aplc', etc)

    Returns:
        list: PINN constraint function, PINN lr annealing function, PINN lr annealing initial values
    """
    from GravNN.Networks.Constraints import (
        no_pinn,
        pinn_A,
        pinn_P,
        pinn_PLC,
        pinn_AP,
        pinn_AL,
        pinn_ALC,
        pinn_APL,
        pinn_APLC,
    )
    from GravNN.Networks.Annealing import (
        no_pinn_anneal,
        pinn_A_anneal,
        pinn_P_anneal,
        pinn_AP_anneal,
        pinn_AL_anneal,
        pinn_ALC_anneal,
        pinn_APL_anneal,
        pinn_APLC_anneal,
        pinn_PLC_anneal,
    )

    # Backwards compatibility (if the value is a function -- take the name of the function then select corresponding values)
    try:
        value = value.__name__
    except:
        pass

    # -1 values in the PINN lr annealing initial values indicates that the value will not get updated.
    return {
        "no_pinn": [no_pinn, no_pinn_anneal, [1.0]],  # scaling ignored
        "pinn_a": [pinn_A, pinn_A_anneal, [1.0]],  # scaling ignored
        "pinn_p": [pinn_P, pinn_P_anneal, [1.0]],
        "pinn_pl": [pinn_P, pinn_P_anneal, [1.0, 1.0]],
        "pinn_plc": [pinn_PLC, pinn_PLC_anneal, [1.0, 1.0, 1.0]],
        "pinn_ap": [pinn_AP, pinn_AP_anneal, [1.0, 1.0]],
        "pinn_al": [pinn_AL, pinn_AL_anneal, [1.0, 1.0]],
        "pinn_alc": [pinn_ALC, pinn_ALC_anneal, [1.0, 1.0, 1.0]],
        "pinn_apl": [pinn_APL, pinn_APL_anneal, [1.0, 1.0, 1.0]],
        "pinn_aplc": [pinn_APLC, pinn_APLC_anneal, [1.0, 1.0, 1.0, 1.0]],
    }[value.lower()]


def _get_PI_constraint(value):
    """Method responsible for getting all variables / methods used in the physics informed constraint.

    Args:
        value (str): PINN constraint name (i.e. 'pinn_A', 'pinn_aplc', etc)

    Returns:
        list: PINN constraint function, PINN lr annealing function, PINN lr annealing initial values
    """
    from GravNN.Networks.Constraints import (
        no_pinn,
        pinn_A,
        pinn_P,
        pinn_PLC,
        pinn_AP,
        pinn_AL,
        pinn_ALC,
        pinn_APL,
        pinn_APLC,
    )

    # Backwards compatibility (if the value is a function -- take the name of the function then select corresponding values)
    try:
        value = value.__name__
    except:
        pass

    # -1 values in the PINN lr annealing initial values indicates that the value will not get updated.
    return {
        "no_pinn": no_pinn,
        "pinn_a": pinn_A,
        "pinn_p": pinn_P,
        "pinn_pl": pinn_P,
        "pinn_plc": pinn_PLC,
        "pinn_ap": pinn_AP,
        "pinn_al": pinn_AL,
        "pinn_alc": pinn_ALC,
        "pinn_apl": pinn_APL,
        "pinn_aplc": pinn_APLC,
    }[value.lower()]

def _get_PI_annealing(value):
    """Method responsible for getting all variables / methods used in the physics informed constraint.

    Args:
        value (str): PINN constraint name (i.e. 'pinn_A', 'pinn_aplc', etc)

    Returns:
        list: PINN constraint function, PINN lr annealing function, PINN lr annealing initial values
    """
    from GravNN.Networks.Annealing import (
        no_pinn_anneal,
        pinn_A_anneal,
        pinn_P_anneal,
        pinn_AP_anneal,
        pinn_AL_anneal,
        pinn_ALC_anneal,
        pinn_APL_anneal,
        pinn_APLC_anneal,
        pinn_PLC_anneal,
    )

    # Backwards compatibility (if the value is a function -- take the name of the function then select corresponding values)
    try:
        value = value.__name__
    except:
        pass

    # -1 values in the PINN lr annealing initial values indicates that the value will not get updated.
    return {
        "no_pinn": no_pinn_anneal,
        "pinn_a": pinn_A_anneal,
        "pinn_p": pinn_P_anneal,
        "pinn_pl": pinn_P_anneal,
        "pinn_plc": pinn_PLC_anneal,
        "pinn_ap": pinn_AP_anneal,
        "pinn_al": pinn_AL_anneal,
        "pinn_alc": pinn_ALC_anneal,
        "pinn_apl": pinn_APL_anneal,
        "pinn_aplc": pinn_APLC_anneal,
    }[value.lower()]

def _get_PI_adaptive_constants(value):
    # Backwards compatibility (if the value is a function -- take the name of the function then select corresponding values)
    try:
        value = value.__name__
    except:
        pass

    # -1 values in the PINN lr annealing initial values indicates that the value will not get updated.
    return {
        "no_pinn": [1.0], 
        "pinn_a": [1.0], 
        "pinn_p": [1.0],
        "pinn_pl": [1.0, 1.0],
        "pinn_plc": [1.0, 1.0, 1.0],
        "pinn_ap": [1.0, 1.0],
        "pinn_al": [1.0, 1.0],
        "pinn_alc": [1.0, 1.0, 1.0],
        "pinn_apl": [1.0, 1.0, 1.0],
        "pinn_aplc": [1.0, 1.0, 1.0, 1.0],
    }[value.lower()]



def _get_network_fcn(name):
    """Helper function to initialize the network used in the PINNGravityModel

    Args:
        name (str): network type (e.g. traditional, sph_traditional, sph_pines_transformer)

    Returns:
        function: network function
    """
    from GravNN.Networks.Networks import (
        TraditionalNet,
        SphericalPinesTraditionalNet,
        SphericalPinesTransformerNet,
        SphericalPinesTraditionalNet_v2,
        SphericalPinesTransformerNet_v2,
        SphericalPinesTransformerNet_v3
    )

    return {
        "traditional": TraditionalNet,
        "sph_pines_traditional": SphericalPinesTraditionalNet,
        "sph_pines_transformer": SphericalPinesTransformerNet,
        "sph_pines_traditional_v2": SphericalPinesTraditionalNet_v2,
        "sph_pines_transformer_v2": SphericalPinesTransformerNet_v2,
        "sph_pines_transformer_v3": SphericalPinesTransformerNet_v3,
    }[name.lower()]


def _get_loss_fcn(name):
    """Helper function to initialize the network loss function

    Args:
        name (str): loss type (e.g. percent, rms, percent_rms)

    Returns:
        function: network function
    """
    from GravNN.Networks.Losses import (
        max_loss,
        percent_summed_loss,
        rms_summed_loss,
        percent_rms_summed_loss,
        percent_avg_loss,
        rms_avg_loss,
        percent_rms_avg_loss,
        avg_percent_summed_rms_loss,
        avg_percent_summed_rms_max_error_loss,
        weighted_mean_percent_loss
    )

    return {
        'max' : max_loss,
        "percent_summed": percent_summed_loss,
        "rms_summed": rms_summed_loss,
        "percent_rms_summed": percent_rms_summed_loss,
        "percent_avg" : percent_avg_loss,
        "rms_avg": rms_avg_loss,
        'percent_rms_avg' : percent_rms_avg_loss,
        'avg_percent_summed_rms' : avg_percent_summed_rms_loss,
        "avg_percent_summed_rms_max_error" : avg_percent_summed_rms_max_error_loss,
        'weighted_mean_percent' : weighted_mean_percent_loss

    }[name.lower()]

def _get_tf_dtype(name):
    import tensorflow as tf

    return {"float16": tf.float16, "float32": tf.float32, "float64": tf.float64}[
        name.lower()
    ]


def populate_config_objects(config):
    """Primary helper function used to convert any strings within the hyperparameter config dictionary
    into the necessary tensorflow objects that will be used in the PINNGravityModel

    Args:
        hparams (dict): dictionary of hyperparameters to overload in the config
        config (dict): dictionary of default hyperparameters

    Returns:
        dict: updated configuration dictionary with proper tensorflow objects
    """
    # config["PINN_constraint_fcn"] = _get_PI_constraint(config["PINN_constraint_fcn"][0])
    # config["network_type"] = [_get_network_fcn(config["network_type"][0])]
    config["dtype"] = [_get_tf_dtype(config["dtype"][0])]

    if "num_units" in config:
        for i in range(1, len(config["layers"][0]) - 1):
            config["layers"][0][i] = config["num_units"][0]

    check_config_combos(config)
    return config


def configure_optimizer(config, mixed_precision):
    """Configure the optimizer to account for mixed precision or not

    Args:
        config (dict): dictionary of hyperparameter and configuration variables
        mixed_precision (module): module containing status mixed precision tf status

    Returns:
        tf.Optimizer: configured optimizer
    """
    optimizer = _get_optimizer(config["optimizer"][0])
    optimizer.learning_rate = config["learning_rate"][0]
    if config["mixed_precision"][0]:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale="dynamic")
    else:
        optimizer.get_scaled_loss = lambda x: x
        optimizer.get_unscaled_gradients = lambda x: x
    return optimizer


def configure_run_args(config, hparams):
    """Helper function to permutate all hyperparameter combinations and load them into a
    multiprocess script.

    Args:
        config (dict): default hyperparameters / configuration variables
        hparams (dict): custom hyperparameters to be loaded into config

    Returns:
        list: list of arguments to be passed into the run function
    """
    keys, values = zip(*hparams.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    args = []
    session_num = 0
    for hparam_inst in permutations_dicts:
        print("--- Starting trial: %d" % session_num)
        print({key: value for key, value in hparam_inst.items()})

        # load the hparams into the config 
        for key, value in hparam_inst.items():
            config[key] = [value]

        args.append((config.copy(),))
        session_num += 1
    return args


def get_gzipped_model_size(model):
    """Get size of gzipped model in bytes

    Args:
        model (PINNGravityModel): custom Tf model

    Returns:
        int: size in bytes
    """
    # Returns size of gzipped model, in bytes.
    _, keras_file = tempfile.mkstemp(".h5")
    model.network.save(keras_file, include_optimizer=False)

    _, zipped_file = tempfile.mkstemp(".zip")
    with zipfile.ZipFile(zipped_file, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)

    return os.path.getsize(zipped_file)


def check_config_combos(config):
    """Helper function used to check if any configurations are incompatible and change them.
    The most prominent error being the number of output nodes exist in a network (must be 1 if PINN gravity
    model).

    Args:
        config (dict): updated configuration and hyperparameter dictionary with compatable arguments
    """
    from GravNN.Networks.Constraints import no_pinn

    if config["PINN_constraint_fcn"][0] != no_pinn:
        if config["layers"][0][-1] != 1:
            print(
                "WARNING: The final layer for a PINN must have one output (the potential, U) -- changing automatically"
            )
            config["layers"][0][-1] = 1
    else:
        if config["layers"][0][-1] != 3:
            config["layers"][0][-1] = 3
            print(
                "WARNING: The final layer for a traditional network must have three outputs (the acceleration vector, a) -- changing automatically"
            )
    if config["network_type"][0].__class__.__name__ == "InceptionNet":
        assert (
            len(config["layers"][0][1]) != 0
        ), "Inception network requires layers with multiple sizes, i.e. [[3, [3,7,11], [3,7,11], 1]]"


def save_df_row(dictionary, df_file):
    """Utility function used to save a configuration / hyperparameter dictionary into a dataframe

    Args:
        dictionary (dict): configuration / hyperparameter dictionary
        df_file (str): path to existing dataframe
    """
    directory = os.path.abspath(".") + "/Data/Dataframes/"
    os.makedirs(directory, exist_ok=True)
    dictionary = dict(sorted(dictionary.items(), key=lambda kv: kv[0]))
    df = pd.DataFrame().from_dict(dictionary).set_index("timetag")
    try:
        df_all = pd.read_pickle(df_file)
        df_all = df_all.append(df)
        df_all.to_pickle(df_file)
    except:
        df.to_pickle(df_file)


def get_df_row(model_id, df_file):
    """Utility function that gets the config information from a df

    Args:
        model_id (float): timetag of a configuration row within df
        df_file (str): path to dataframe

    Returns:
        [type]: [description]
    """
    original_df = pd.read_pickle(df_file)
    config = original_df[model_id == original_df["id"]].to_dict()
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
    timestamp = pd.to_datetime(model_id, unit="D", origin="julian").round("s").ctime()
    entries.update({"timetag": [timestamp]})
    dictionary = dict(sorted(entries.items(), key=lambda kv: kv[0]))
    df = pd.DataFrame.from_dict(dictionary).set_index("timetag")
    original_df = original_df.combine_first(df)
    original_df.update(df)  # , sort=True) # join, merge_ordered also viable
    if save:
        original_df.to_pickle(df_file)
    return original_df


def format_config(config):
    new_config = deepcopy(config)
    new_config['planet'] = [new_config['planet'][0].__class__.__name__]
    new_config['distribution'] = [new_config['distribution'][0].__name__]
    new_config['network_type'] = [new_config['network_type'][0].__name__]
    new_config['PINN_constraint_fcn'] = [new_config['PINN_constraint_fcn'][0].__name__]
    new_config['x_transformer'] = [new_config['x_transformer'][0].__class__.__name__]
    new_config['a_transformer'] = [new_config['a_transformer'][0].__class__.__name__]
    new_config['u_transformer'] = [new_config['u_transformer'][0].__class__.__name__]
    new_config['a_bar_transformer'] = [new_config['a_bar_transformer'][0].__class__.__name__]
    new_config['dummy_transformer'] = [new_config['dummy_transformer'][0].__class__.__name__]
    new_config['grav_file'] = [new_config['grav_file'][0].split("/")[-1]]
    new_config['deg_removed'] = [new_config.get('deg_removed', ['None'])[0]]
    new_config['remove_point_mass'] = [new_config.get('remove_point_mass', ['None'])[0]]
    return new_config

def print_config(original_config):
    config = format_config(original_config)
    data_keys = ['planet', 'distribution', 'grav_file',  'deg_removed', 'remove_point_mass',
        'N_dist', 'N_train', 'N_val', 
        'radius_min', 'radius_max', 'scale_by', 
        'acc_noise', 'override', 'seed' ,
        'x_transformer', 'a_transformer', 'u_transformer', 
        'a_bar_transformer',
        ]
    init(autoreset=True)

    print(Back.BLUE + Fore.BLACK + "Data Hyperparams")
    for key in data_keys:
        print(Fore.BLUE +  "{:<20}\t".format(key) + Fore.WHITE + " {:<15}".format(str(config.get(key, ['None'])[0])))
        del config[key]
    print("\n")
    network_keys = ['PINN_constraint_fcn', 'network_type', 'layers', 
        'activation', 'epochs',  'learning_rate', 
        'batch_size', 'initializer',
        'optimizer', 'dropout', 'normalization_strategy', 
        'mixed_precision', 'init_file', 'id'
        ]
    print(Back.RED + Fore.BLACK + "Network Hyperparams")
    for key in network_keys:
        print(Fore.RED + "{:<20}\t".format(key) + Fore.WHITE + " {:<15}".format(str(config[key][0])))
        del config[key]
    print("\n")

    scheduler_keys = ['schedule_type', 'lr_anneal', 'min_delta', 
        'min_lr', 'patience', 'decay_rate', 'beta'
        ]
    print(Back.YELLOW + Fore.BLACK + "Learning Rate Scheduler Hyperparams")
    for key in scheduler_keys:
        print(Fore.YELLOW +  "{:<20}\t".format(key) + Fore.WHITE + " {:<15}".format(str(config[key][0])))
        del config[key]
    print("\n")

    stats_keys = ['size', 'params', 'time_delta']
    print(Back.GREEN + Fore.BLACK + "Statistics")
    for key in stats_keys:
        print(Fore.GREEN +  "{:<20}\t".format(key) + Fore.WHITE + " {:<15}".format(str(config[key][0])))
        del config[key]
    print(Fore.GREEN + "{:<20}\t".format("Final Loss") + Fore.WHITE + "{:<20}".format(config['history'][0]['loss'][-1]))
    print(Fore.GREEN + "{:<20}\t".format("Final Val Loss") + Fore.WHITE + "{:<20}".format(config['history'][0]['val_loss'][-1]))
    print("\n")

    print(Back.MAGENTA + Fore.BLACK + "Miscellaneous Hyperparams")
    for key,value in config.items():
        if key == 'history':
            continue
        print(Fore.MAGENTA +  "{:<20}\t".format(key) + Fore.WHITE + " {:<15}".format(str(config[key][0])))

    deinit()




def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    import tensorflow as tf
    for i in range(0, len(lst), n):
        yield tf.data.Dataset.from_tensor_slices(lst[i:i + n])
