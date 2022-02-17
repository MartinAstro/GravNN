"""Collection of functions used to extract, format, and preprocess training data for the networks"""

import numpy as np
import tensorflow as tf
import copy
from scipy.spatial.transform import Rotation
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from GravNN.GravityModels.PointMass import PointMass
from GravNN.GravityModels.Polyhedral import Polyhedral, get_poly_data
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.Preprocessors.DummyScaler import DummyScaler
from GravNN.Networks.Constraints import *
from GravNN.Support.transformations import project_acceleration, cart2sph, sphere2cart
from sklearn.preprocessing import MinMaxScaler


def print_stats(data, name):
    """Evaluate the training data statistics. Useful
    to determine if the preprocessing transformation worked
    properly.

    Args:
        data (np.array): data to be evaluated
        name (str): descriptor of data
    """
    print(
        "(%s) \t Max: %.4f, Min: %.4f, Range: %.4f \n \tAvg: %.4f, Std: %.4f, Median: %.4f"
        % (
            name,
            np.max(data),
            np.min(data),
            np.max(data) - np.min(data),
            np.mean(data),
            np.std(data),
            np.median(data),
        )
    )


def compute_input_layer_normalization_constants(config):
    """Function responsible for determining how to normalize the spherical coordinates
    before entering the network. Currently the tensorflow graph accepts a normalized cartesian vector
    input, which is then (within the graph) transformed to spherical coordinates. The theta and phi components
    grossly exceed the [-1,1] bounds if left unnormalized, so this function identifies those mins and max values
    for normalization purposes within the graph. 

    .. Note:: In the PinesSphericalNet these constants are generally irrelevant given the 4D coordinate system
    in which all parameters exist naturally between -1,1. Consequently this is only used for the SphTraditionalNet
    with 3D coordinates. 

    Args:
        config (dict): hyperparameters and configuration variables for the TF model.
    """
    if config["skip_normalization"][0]:
        return

    trajectory = config["distribution"][0](
        config["planet"][0],
        [config["radius_min"][0], config["radius_max"][0]],
        config["N_dist"][0],
        **config
    )
    get_analytic_data_fcn = config['gravity_data_fcn'][0]
    
    x_unscaled, a_unscaled, u_unscaled = get_analytic_data_fcn(
        trajectory, config["grav_file"][0], **config
    )

    # Transform into cartesian normalized (i.e unit vectors)
    x_unscaled = config["x_transformer"][0].transform(x_unscaled)

    # Convert to spherical coordinates
    x_unscaled = cart2sph(x_unscaled)
    x_unscaled[:, 1:3] = np.deg2rad(x_unscaled[:, 1:3])

    x_train, a_train, u_train, x_val, a_val, u_val = training_validation_split(
        x_unscaled, a_unscaled, u_unscaled, config["N_train"][0], config["N_val"][0]
    )

    a_train = a_train + config["acc_noise"][0] * np.std(a_train) * np.random.randn(
        a_train.shape[0], a_train.shape[1]
    )

    # Determine scalers for the normalized spherical coordinates

    # Scale the radial component independently
    r_transformer = MinMaxScaler(feature_range=(-1, 1))
    x_train[:, 0:1] = r_transformer.fit_transform(x_train[:, 0:1])

    # Also scale the angles together
    angles_transformer = MinMaxScaler(feature_range=(-1, 1))
    x_train[:, 1:3] = angles_transformer.fit_transform(x_train[:, 1:3])

    config["norm_scalers"] = [
        np.concatenate([r_transformer.scale_, angles_transformer.scale_])
    ]
    config["norm_mins"] = [np.concatenate([r_transformer.min_, angles_transformer.min_])]
    return


def get_raw_data(config):
    """Function responsible for getting the raw training data (without
    any preprocessing). This may include concatenating an "extra" training
    data distribution defined within config.

    Args:
        config (dict): hyperparameters and configuration variables for TF Model

    Returns:
        tuple: x,a,u training and validation data
    """
    planet = config["planet"][0]
    radius_bounds = [config["radius_min"][0], config["radius_max"][0]]
    N_dist = config["N_dist"][0]
    grav_file = config["grav_file"][0]

    trajectory = config["distribution"][0](planet, radius_bounds, N_dist, **config)
    get_analytic_data_fcn = config['gravity_data_fcn'][0]
    
    x_unscaled, a_unscaled, u_unscaled = get_analytic_data_fcn(
        trajectory, grav_file, **config
    )

    if "extra_distribution" in config:
        extra_radius_bounds = [
            config["extra_radius_min"][0],
            config["extra_radius_max"][0],
        ]
        extra_N_dist = config["extra_N_dist"][0]

        extra_trajectory = config["extra_distribution"][0](
            planet, extra_radius_bounds, extra_N_dist, **config
        )
        extra_x_unscaled, extra_a_unscaled, extra_u_unscaled = get_analytic_data_fcn(
            extra_trajectory, grav_file, **config
        )

    # This condition is for meant to correct for when gravity models didn't always have the proper sign of the potential.
    # TODO: This should be removed prior to production.
    deg_removed = config.get("deg_removed", -1)
    # if part of the model is removed (i.e. point mass) it is reasonable for some of the potential to be > 0.0, so only rerun if there is no degree removed.
    if np.max(u_unscaled) > 0.0 and deg_removed == -1:
        exit("ERROR: This pickled acceleration/potential pair was generated \
              when the potential had a wrong sign. \n You must overwrite the data!")

    x_train, a_train, u_train, x_val, a_val, u_val = training_validation_split(
        x_unscaled, a_unscaled, u_unscaled, config["N_train"][0], config["N_val"][0], random_state=config.get('seed', [42])[0]
    )

    if "extra_distribution" in config:
        (
            extra_x_train,
            extra_a_train,
            extra_u_train,
            extra_x_val,
            extra_a_val,
            extra_u_val,
        ) = training_validation_split(
            extra_x_unscaled,
            extra_a_unscaled,
            extra_u_unscaled,
            config["extra_N_train"][0],
            config["extra_N_val"][0],
        )

        x_train = np.concatenate([x_train, extra_x_train])
        a_train = np.concatenate([a_train, extra_a_train])
        u_train = np.concatenate([u_train, extra_u_train])

        x_val = np.concatenate([x_val, extra_x_val])
        a_val = np.concatenate([a_val, extra_a_val])
        u_val = np.concatenate([u_val, extra_u_val])

    return x_train, a_train, u_train, x_val, a_val, u_val


def get_preprocessed_data(config):
    """Function responsible for normalizing the training data. Possible options include
    normalizing by the bounds of the acceleration, the potential, neither, or in a manner that non-
    dimensionalizes the equations (this amounts to scaling the acceleration in proportion to
    the more fundamental scalar potential and position). Recommend normalizing by non-dim
    in most circumstances for which the underlying network is a PINN."""
    x_train, a_train, u_train, x_val, a_val, u_val = get_raw_data(config)

    print_stats(x_train, "Position")
    print_stats(a_train, "Acceleration")
    print_stats(u_train, "Potential")

    a_mag = np.linalg.norm(a_train, axis=1).reshape(len(a_train),1)
    a_unit = np.random.uniform(-1,1, size=np.shape(a_train))
    a_unit = a_unit / np.linalg.norm(a_unit, axis=1).reshape(len(a_unit), 1)
    a_error = config['acc_noise'][0]*a_mag*a_unit # 10% of the true magnitude 
    a_train = a_train + a_error

    # Preprocessing
    x_transformer = config["x_transformer"][0]
    a_transformer = config["a_transformer"][0]
    u_transformer = config["u_transformer"][0]

    a_bar_transformer = copy.deepcopy(config["a_transformer"][0])

    if config["scale_by"][0] == "a":
        # Scale (a,u) with a_transformer
        x_train = x_transformer.fit_transform(x_train)
        a_train = a_transformer.fit_transform(a_train)
        u_train = a_transformer.transform(np.repeat(u_train, 3, axis=1))[:, 0].reshape(
            (-1, 1)
        )

        x_val = x_transformer.transform(x_val)
        a_val = a_transformer.transform(a_val)
        u_val = a_transformer.transform(np.repeat(u_val, 3, axis=1))[:, 0].reshape(
            (-1, 1)
        )
        u_transformer = a_transformer
    elif config["scale_by"][0] == "u":
        # Scale (a,u) with u_transformer
        x_train = x_transformer.fit_transform(x_train)
        u_train = u_transformer.fit_transform(np.repeat(u_train, 3, axis=1))[
            :, 0
        ].reshape((-1, 1))
        a_train = u_transformer.transform(a_train)

        x_val = x_transformer.transform(x_val)
        a_val = u_transformer.transform(a_val)
        u_val = u_transformer.transform(np.repeat(u_val, 3, axis=1))[:, 0].reshape(
            (-1, 1)
        )
        a_transformer = u_transformer
    elif config["scale_by"][0] == "non_dim":
        """
        x_bar = (x-x0)/xs
        u_bar = (u-u0)/us

        The acceleration is a byproduct of the potential so

        a_bar = du_bar/dx_bar

        but this equates to

        a_bar = d((u-u0)/us)/dx_bar = d(u/us)/dx_bar = 1/us*du/dx_bar = 1/us *du/(d((x-x0)/xs)) = 1/us * du/d(x/xs) = xs/us * du/dx = xs/us * a

        such that a_bar exists in [-1E2, 1E2] rather than [-1,1]
        """

        # Designed to make position, acceleration, and potential all exist between [-1,1]
        x_train = x_transformer.fit_transform(x_train)
        u_train = u_transformer.fit_transform(np.repeat(u_train, 3, axis=1))[
            :, 0
        ].reshape((-1, 1))

        # Scale the acceleration by the potential
        scaler = 1 / (x_transformer.scale_ / u_transformer.scale_)
        a_bar = a_transformer.fit_transform(a_train, scaler=scaler)  # this is a_bar
        a_bar_transformer.fit(a_bar)
        a_train = a_bar  # this should be on the order of [-100, 100]
        config["a_bar_transformer"] = [a_bar_transformer]

        u3vec = np.repeat(u_val, 3, axis=1)

        x_val = x_transformer.transform(x_val)
        a_val = a_transformer.transform(a_val)
        u_val = u_transformer.transform(u3vec)[:, 0].reshape((-1, 1))

        ref_r_vec = np.array([[config['ref_radius'][0], 0, 0]])
        ref_r_vec = x_transformer.transform(ref_r_vec)
        config['ref_radius'] = [ref_r_vec[0,0].astype(np.float32)]

    elif config["scale_by"][0] == "none":
        x_transformer = config["dummy_transformer"][0]
        a_transformer = config["dummy_transformer"][0]
        u_transformer = config["dummy_transformer"][0]

    print_stats(x_train, "Scaled Position")
    print_stats(a_train, "Scaled Acceleration")
    print_stats(u_train, "Scaled Potential")

    laplace_train = np.zeros((np.shape(u_train)))
    laplace_val = np.zeros((np.shape(u_val)))

    curl_train = np.zeros((np.shape(a_train)))
    curl_val = np.zeros((np.shape(a_val)))

    transformers = {
        "x": x_transformer,
        "a": a_transformer,
        "u": u_transformer,
        "a_bar": a_bar_transformer,
    }

    return (
        (x_train, u_train, a_train, laplace_train, curl_train),
        (x_val, u_val, a_val, laplace_val, curl_val),
        transformers,
    )


def configure_dataset(train_data, val_data, config):
    """Function that partitions the training data to include only that
    which is required for the use PINN constraint. I.e. if using the
    AP constraint, there is no need to send extra vectors for L and C
    onto the device (GPU) and slow calculations."""
    x_train, u_train, a_train, laplace_train, curl_train = train_data
    x_val, u_val, a_val, laplace_val, curl_val = val_data
    pinn_constraint_fcn = config["PINN_constraint_fcn"][0]

    # Decide to train with potential or not
    # TODO: Modify which variables are added to the state to speed up training and minimize memory footprint.
    if pinn_constraint_fcn == no_pinn:
        y_train = np.hstack([a_train])
        y_val = np.hstack([a_val])

    # Just use acceleration
    elif pinn_constraint_fcn == pinn_A:
        y_train = np.hstack([a_train])
        y_val = np.hstack([a_val])

    # Just use acceleration
    elif pinn_constraint_fcn == pinn_P:
        y_train = np.hstack([u_train])
        y_val = np.hstack([u_val])

    # Potential + 2nd order constraints
    elif pinn_constraint_fcn == pinn_PL:
        y_train = np.hstack([u_train, laplace_train])
        y_val = np.hstack([u_val, laplace_val])
    elif pinn_constraint_fcn == pinn_PLC:
        y_train = np.hstack([u_train, laplace_train, curl_train])
        y_val = np.hstack([u_val, laplace_val, curl_val])

    # Accelerations + 2nd order constraints
    elif pinn_constraint_fcn == pinn_AL:
        y_train = np.hstack([a_train, laplace_train])
        y_val = np.hstack([a_val, laplace_val])
    elif pinn_constraint_fcn == pinn_ALC:
        y_train = np.hstack([a_train, laplace_train, curl_train])
        y_val = np.hstack([a_val, laplace_val, curl_val])

    # Accelerations + Potential + 2nd order constraints
    elif pinn_constraint_fcn == pinn_AP:
        y_train = np.hstack([u_train, a_train])
        y_val = np.hstack([u_val, a_val])
    elif pinn_constraint_fcn == pinn_APL:
        y_train = np.hstack([u_train, a_train, laplace_train])
        y_val = np.hstack([u_val, a_val, laplace_val])
    elif pinn_constraint_fcn == pinn_APLC:
        y_train = np.hstack([u_train, a_train, laplace_train, curl_train])
        y_val = np.hstack([u_val, a_val, laplace_val, curl_val])

    else:
        exit("No PINN Constraint Selected!")

    dataset = generate_dataset(
        x_train, y_train, config["batch_size"][0], dtype=config["dtype"][0]
    )
    val_dataset = generate_dataset(
        x_val, y_val, config["batch_size"][0], dtype=config["dtype"][0]
    )

    return dataset, val_dataset


def generate_dataset(x, y, batch_size, dtype=None):
    """Function which takes numpy arrays and converts
    them into a tensorflow Dataset -- a much faster
    type for training."""
    if dtype is None:
        x = x.astype("float32")
        y = y.astype("float32")
    else:
        if dtype == tf.float32:
            x = x.astype("float32")
            y = y.astype("float32")
        elif dtype == tf.float64:
            x = x.astype("float64")
            y = y.astype("float64")
        else:
            exit("No dtype specified")
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # dataset = dataset.shuffle(1000, seed=1234)
    dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()

    # Why Cache is Impt: https://stackoverflow.com/questions/48240573/why-is-tensorflows-tf-data-dataset-shuffle-so-slow
    return dataset


def single_training_validation_split(X, N_train, N_val, random_state=42):
    """Function responsible for splitting the variable into separate training
    and validation sets"""
    X = shuffle(X, random_state=random_state)
    X_train = X[:N_train]
    X_val = X[N_train:N_train+N_val]
    return X_train, X_val


def training_validation_split(X, Y, Z, N_train, N_val, random_state=42):
    """Function which automates splitting the training and validation data
    for all variables (typically position, acceleration, and potential)"""
    X_train, X_val = single_training_validation_split(
        X, N_train, N_val, random_state=random_state
    )
    Y_train, Y_val = single_training_validation_split(
        Y, N_train, N_val, random_state=random_state
    )
    Z_train, Z_val = single_training_validation_split(
        Z, N_train, N_val, random_state=random_state
    )

    return X_train, Y_train, Z_train, X_val, Y_val, Z_val
