import itertools
import logging
import os
import pickle
import tempfile
import zipfile
from copy import deepcopy

import pandas as pd
from colorama import Fore, deinit, init
from colorama.ansi import Back

import GravNN


def configure_tensorflow(hparams):
    """Custom tensorflow import that configures proper flags, path settings,
    seeds, etc.

    Returns:
        module: Tensorflow as tf
    """
    set_tf_env_flags()
    tf = set_tf_expand_memory()
    tf.keras.backend.clear_session()
    tf.random.set_seed(hparams["seed"][0])
    tf.config.run_functions_eagerly(hparams["eager"][0])
    mixed_precision = set_mixed_precision() if hparams["mixed_precision"][0] else None

    return tf, mixed_precision


def set_tf_env_flags():
    """Add CUDA library to path (assumes using v10.1) and configure GPU and XLA."""
    import os

    os.environ["PATH"] += (
        os.pathsep
        + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\"
        + "extras\\CUPTI\\lib64"
    )
    os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/jit/flags.cc
    # #L82-L142 list of XLA flags
    os.environ[
        "TF_XLA_FLAGS"
    ] = "--tf_xla_enable_xla_devices --tf_xla_cpu_global_jit --tf_xla_auto_jit=1"


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
    """Method used to configure mixed precision settings. This allows for faster
    training times for non-physics informed neural networks.

    .. warn:: Do not configure mixed precision when training PINNs. Because gradients
    of the network are embedded within the loss function, the cruder precision can cause
     convergence issues.

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
    elif "sgd" in name.lower():
        name = "wadam"
    else:
        pass

    return {
        "sgd": tf.keras.optimizers.SGD(),
        "adagrad": tf.keras.optimizers.Adagrad(),
        "adadelta": tf.keras.optimizers.Adadelta(),
        "rmsprop": tf.keras.optimizers.RMSprop(),
        "adam": tf.keras.optimizers.Adam(),
        # "wadam": tf.keras.optimizers.experimental.AdamW()
    }[name.lower()]


def _get_loss_fcn(name):
    """Helper function to initialize the network loss function

    Args:
        name (str): loss type (e.g. percent, rms, percent_rms)

    Returns:
        function: network function
    """
    from GravNN.Networks.Losses import (
        avg_percent_summed_rms_loss,
        avg_percent_summed_rms_max_error_loss,
        max_loss,
        percent_avg_loss,
        percent_rms_avg_loss,
        percent_rms_summed_loss,
        percent_summed_loss,
        rms_avg_loss,
        rms_summed_loss,
        weighted_mean_percent_loss,
    )

    return {
        "max": max_loss,
        "percent_summed": percent_summed_loss,
        "rms_summed": rms_summed_loss,
        "percent_rms_summed": percent_rms_summed_loss,
        "percent_avg": percent_avg_loss,
        "rms_avg": rms_avg_loss,
        "percent_rms_avg": percent_rms_avg_loss,
        "avg_percent_summed_rms": avg_percent_summed_rms_loss,
        "avg_percent_summed_rms_max_error": avg_percent_summed_rms_max_error_loss,
        "weighted_mean_percent": weighted_mean_percent_loss,
    }[name.lower()]


def _get_tf_dtype(name):
    import tensorflow as tf

    return {"float16": tf.float16, "float32": tf.float32, "float64": tf.float64}[
        name.lower()
    ]


def populate_config_objects(config):
    """Primary helper function used to convert any strings within the hyperparameter
    config dictionary into the necessary tensorflow objects that will be used in the
    PINNGravityModel

    Args:
        hparams (dict): dictionary of hyperparameters to overload in the config
        config (dict): dictionary of default hyperparameters

    Returns:
        dict: updated configuration dictionary with proper tensorflow objects
    """

    config["dtype"] = [_get_tf_dtype(config["dtype"][0])]

    if "num_units" in config:
        for i in range(1, len(config["layers"][0]) - 1):
            config["layers"][0][i] = config["num_units"][0]
        logging.info("Changed Layers to: " + str(config["layers"][0]))

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


def permutate_dict(dictionary):
    if len(dictionary) == 0:
        return [dictionary]
    keys, values = zip(*dictionary.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return permutations_dicts


def configure_run_args(config, hparams):
    """Helper function to permutate all hyperparameter combinations and load them into a
    multiprocess script.

    Args:
        config (dict): default hyperparameters / configuration variables
        hparams (dict): custom hyperparameters to be loaded into config

    Returns:
        list: list of arguments to be passed into the run function
    """
    permutations_dicts = permutate_dict(hparams)
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
    """Helper function used to check if any configurations are incompatible and change
    them. The most prominent error being the number of output nodes exist in a network
    (must be 1 if PINN gravity model).
    Args:
        config (dict): updated configuration and hyperparameter dictionary with
        compatable arguments
    """
    from GravNN.Networks.Constraints import pinn_00

    if config["PINN_constraint_fcn"][0] != pinn_00:
        if config["layers"][0][-1] != 1:
            print(
                "WARNING: The final layer for a PINN must have one output \
                    (the potential, U) -- changing automatically",
            )
            config["layers"][0][-1] = 1
    else:
        if config["layers"][0][-1] != 3:
            config["layers"][0][-1] = 3
            print(
                "WARNING: The final layer for a traditional network must have three \
                    outputs (the acceleration vector, a) -- changing automatically",
            )
    if config["network_type"][0].__class__.__name__ == "InceptionNet":
        assert (
            len(config["layers"][0][1]) != 0
        ), "Inception network requires layers with multiple sizes, i.e. [[3, [3,7,11], \
            [3,7,11], 1]]"

    # from GravNN.CelestialBodies.Asteroids import Asteroid
    # from GravNN.GravityModels.Polyhedral import Polyhedral

    # planet = config["planet"][0]
    # if isinstance(planet, Asteroid):
    #     obj_file = config["obj_file"][0]
    #     model = Polyhedral(planet, obj_file)
    #     volume = trimesh.load_mesh(model.obj_file).volume * 1e9
    #     G = 6.67430 * 10**-11
    #     config["mu"] = [volume * planet.density * G]
    #     print("Modified Mu to reflect shape file: ", config["mu"])


def save_df_row(dictionary, df_file):
    """Utility function used to save a configuration / hyperparameter dictionary into a
     dataframe

    Args:
        dictionary (dict): configuration / hyperparameter dictionary
        df_file (str): path to existing dataframe
    """
    directory = os.path.dirname(df_file)
    os.makedirs(directory, exist_ok=True)
    dictionary = dict(sorted(dictionary.items(), key=lambda kv: kv[0]))
    df = pd.DataFrame().from_dict(dictionary).set_index("timetag")
    try:
        df_all = pd.read_pickle(df_file)
        df_all = df_all.append(df)
        df_all.to_pickle(df_file)
        print("Saved to existing dataframe:", df_file)
    except Exception:
        df.to_pickle(df_file)
        print("Saved to new dataframe:", df_file)


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
        df_file (any): Either the path used to load the df (slow) or df itself (fast)
        entries (series): The series to update in the df
        save (bool, optional): Save the dataframe immediately after updating (slow).

    Returns:
        DataFrame: The updated dataframe
    """
    if type(df_file) == str:
        original_df = pd.read_pickle(df_file)
    else:
        original_df = df_file
    timestamp = pd.to_datetime(model_id, unit="D", origin="julian").to_julian_date()
    entries.update({"timetag": [timestamp]})
    dictionary = dict(sorted(entries.items(), key=lambda kv: kv[0]))
    df = pd.DataFrame.from_dict(dictionary).set_index("timetag")
    original_df = original_df.combine_first(df)
    original_df.update(df)  # , sort=True) # join, merge_ordered also viable
    if save:
        original_df.to_pickle(df_file)
    return original_df


def get_history(model_id):
    network_dir = os.path.dirname(GravNN.__file__) + f"/../Data/Networks/{model_id}/"
    with open(network_dir + "history.data", "rb") as f:
        history = pickle.load(f)
    return history


def format_config(config):
    new_config = deepcopy(config)
    new_config["planet"] = [new_config["planet"][0].__class__.__name__]
    new_config["distribution"] = [new_config["distribution"][0].__name__]
    new_config["gravity_data_fcn"] = [new_config["gravity_data_fcn"][0].__name__]
    new_config["x_transformer"] = [new_config["x_transformer"][0].__class__.__name__]
    new_config["a_transformer"] = [new_config["a_transformer"][0].__class__.__name__]
    new_config["u_transformer"] = [new_config["u_transformer"][0].__class__.__name__]
    new_config["a_bar_transformer"] = [
        new_config["a_bar_transformer"][0].__class__.__name__,
    ]
    new_config["dummy_transformer"] = [
        new_config["dummy_transformer"][0].__class__.__name__,
    ]
    new_config["obj_file"] = [new_config["obj_file"][0].split("/")[-1]]
    new_config["deg_removed"] = [new_config.get("deg_removed", ["None"])[0]]
    new_config["remove_point_mass"] = [new_config.get("remove_point_mass", ["None"])[0]]
    return new_config


def print_config(original_config):
    config = format_config(original_config)
    data_keys = [
        "planet",
        "distribution",
        "obj_file",
        "deg_removed",
        "remove_point_mass",
        "N_dist",
        "N_train",
        "N_val",
        "radius_min",
        "radius_max",
        "scale_by",
        "acc_noise",
        "override",
        "seed",
        "x_transformer",
        "a_transformer",
        "u_transformer",
        "a_bar_transformer",
        "gravity_data_fcn",
        "mu",
        "mu_non_dim",
    ]
    init(autoreset=True)

    print(Back.BLUE + Fore.BLACK + "Data Hyperparams")
    for key in data_keys:
        print(
            Fore.BLUE
            + "{:<20}\t".format(key)
            + Fore.WHITE
            + " {:<15}".format(str(config.get(key, ["None"])[0])),
        )
        del config[key]
    print("\n")
    network_keys = [
        "PINN_constraint_fcn",
        "network_arch",
        "layers",
        "activation",
        "epochs",
        "learning_rate",
        "batch_size",
        "initializer",
        "optimizer",
        "dropout",
        "mixed_precision",
        "init_file",
        "blend_potential",
        "final_layer_initializer",
        "fuse_models",
        "preprocessing",
        "scale_nn_potential",
        "tanh_k",
        "trainable_tanh",
        "jit_compile",
        "loss_fcns",
        "eager",
        "enforce_bc",
    ]
    print(Back.RED + Fore.BLACK + "Network Hyperparams")
    for key in network_keys:
        print(
            Fore.RED
            + "{:<20}\t".format(key)
            + Fore.WHITE
            + " {:<15}".format(str(config[key][0])),
        )
        del config[key]
    print("\n")

    scheduler_keys = [
        "schedule_type",
        "lr_anneal",
        "min_delta",
        "min_lr",
        "patience",
        "decay_rate",
        "beta",
    ]
    print(Back.YELLOW + Fore.BLACK + "Learning Rate Scheduler Hyperparams")
    for key in scheduler_keys:
        print(
            Fore.YELLOW
            + "{:<20}\t".format(key)
            + Fore.WHITE
            + " {:<15}".format(str(config[key][0])),
        )
        del config[key]
    print("\n")

    stats_keys = ["size", "params"]
    print(Back.GREEN + Fore.BLACK + "Statistics")
    for key in stats_keys:
        print(
            Fore.GREEN
            + "{:<20}\t".format(key)
            + Fore.WHITE
            + " {:<15}".format(str(config[key][0])),
        )
        del config[key]
    history = get_history(config["id"][0])
    print(
        Fore.GREEN
        + "{:<20}\t".format("Final Loss")
        + Fore.WHITE
        + "{:<20}".format(history["loss"][-1]),
    )
    print(
        Fore.GREEN
        + "{:<20}\t".format("Final Val Loss")
        + Fore.WHITE
        + "{:<20}".format(history["val_loss"][-1]),
    )
    print("\n")

    print(Back.MAGENTA + Fore.BLACK + "Miscellaneous Hyperparams")
    for key, value in config.items():
        if key == "history":
            continue
        print(
            Fore.MAGENTA
            + "{:<20}\t".format(key)
            + Fore.WHITE
            + " {:<15}".format(str(config[key][0])),
        )

    deinit()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, lst.shape[0], n):
        yield lst[i : i + n]


def get_absolute_path(file_path, make_path=False):
    # Check if the file path is absolute
    if os.path.isabs(file_path):
        abs_path = file_path
    else:
        # Check if the path was relative and convert it to absolute path
        abs_path = os.path.abspath(file_path)

    # Check if the file path is just a filename
    if not os.path.exists(abs_path):
        # Assuming the file is in the relative path
        relative_path = os.path.join(os.getcwd(), file_path)
        if os.path.exists(relative_path):
            abs_path = relative_path
        else:
            if make_path is False:
                raise FileNotFoundError(f"The file '{file_path}' does not exist.")
            else:
                os.makedirs(abs_path, exist_ok=True)
    return abs_path
