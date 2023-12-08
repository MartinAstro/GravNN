"""Collection of functions used to extract, format, and preprocess training
     data for the networks"""

import copy
import sys

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from GravNN.CelestialBodies.Planets import Planet
from GravNN.Networks.Constraints import *
from GravNN.Preprocessors.DummyScaler import DummyScaler
from GravNN.Support.PathTransformations import make_windows_path_posix


def print_stats(data, name):
    """Evaluate the training data statistics. Useful
    to determine if the preprocessing transformation worked
    properly.

    Args:
        data (np.array): data to be evaluated
        name (str): descriptor of data
    """
    print(
        "(%s) \t Max: %.4f, Min: %.4f, Range: %.4f \n \
              \t Avg: %.4f, Std: %.4f, Median: %.4f"
        % (
            name,
            np.max(data),
            np.min(data),
            np.max(data) - np.min(data),
            np.mean(data),
            np.std(data),
            np.median(data),
        ),
    )


def add_error(data_dict, percent_noise):
    a_train = data_dict["a_train"]

    a_mag = np.linalg.norm(a_train, axis=1).reshape(len(a_train), 1)
    a_unit = np.random.uniform(-1, 1, size=np.shape(a_train))
    a_unit = a_unit / np.linalg.norm(a_unit, axis=1).reshape(len(a_unit), 1)
    a_error = percent_noise * a_mag * a_unit  # 10% of the true magnitude
    a_train = a_train + a_error
    data_dict["a_train"] = a_train

    return data_dict


def scale_by_acceleration(data_dict, config):
    x_transformer = config["x_transformer"][0]
    a_transformer = config["a_transformer"][0]
    u_transformer = config["u_transformer"][0]
    a_bar_transformer = config["a_transformer"][0]

    # Scale (a,u) with a_transformer
    u_train_vals = np.repeat(data_dict["u_train"].reshape((-1, 1)), 3, axis=1)
    u_valid_vals = np.repeat(data_dict["u_val"].reshape((-1, 1)), 3, axis=1)

    x_train = x_transformer.fit_transform(data_dict["x_train"])
    a_train = a_transformer.fit_transform(data_dict["a_train"])
    u_train = a_transformer.transform(u_train_vals)[:, 0].reshape((-1, 1))

    x_val = x_transformer.transform(data_dict["x_val"])
    a_val = a_transformer.transform(data_dict["a_val"])
    u_val = a_transformer.transform(u_valid_vals)[:, 0].reshape((-1, 1))
    u_transformer = a_transformer

    data_dict = {
        "x_train": x_train,
        "a_train": a_train,
        "u_train": u_train,
        "x_val": x_val,
        "a_val": a_val,
        "u_val": u_val,
    }

    transformers = {
        "x": x_transformer,
        "a": a_transformer,
        "u": u_transformer,
        "a_bar": a_bar_transformer,
    }
    return data_dict, transformers


def scale_by_potential(data_dict, config):
    x_transformer = config["x_transformer"][0]
    a_transformer = config["a_transformer"][0]
    u_transformer = config["u_transformer"][0]
    a_bar_transformer = config["a_transformer"][0]

    # Scale (a,u) with u_transformer
    u_train_vals = np.repeat(data_dict["u_train"], 3, axis=1)
    u_valid_vals = np.repeat(data_dict["u_val"], 3, axis=1)

    x_train = x_transformer.fit_transform(data_dict["x_train"])
    u_train = u_transformer.fit_transform(u_train_vals)[:, 0].reshape((-1, 1))
    a_train = u_transformer.transform(data_dict["a_train"])

    x_val = x_transformer.transform(data_dict["x_val"])
    a_val = u_transformer.transform(data_dict["a_val"])
    u_val = u_transformer.transform(u_valid_vals)[:, 0].reshape((-1, 1))
    a_transformer = u_transformer

    data_dict = {
        "x_train": x_train,
        "a_train": a_train,
        "u_train": u_train,
        "x_val": x_val,
        "a_val": a_val,
        "u_val": u_val,
    }

    transformers = {
        "x": x_transformer,
        "a": a_transformer,
        "u": u_transformer,
        "a_bar": a_bar_transformer,
    }
    return data_dict, transformers


def scale_by_non_dimensional(data_dict, config):
    """
    x_bar = (x-x0)/xs
    u_bar = (u-u0)/us

    The acceleration is a byproduct of the potential so

    a_bar = du_bar/dx_bar

    but this equates to

    a_bar = d((u-u0)/us)/dx_bar = d(u/us)/dx_bar = 1/us*du/dx_bar
          = 1/us *du/(d((x-x0)/xs)) = 1/us * du/d(x/xs) = xs/us * du/dx = xs/us * a

    such that a_bar exists in [-1E2, 1E2] rather than [-1,1]
    """
    x_transformer = config["x_transformer"][0]
    a_transformer = config["a_transformer"][0]
    u_transformer = config["u_transformer"][0]
    # a_bar_transformer = config["a_transformer"][0]

    u_train_vals = np.repeat(data_dict["u_train"].reshape((-1, 1)), 3, axis=1)
    u_valid_vals = np.repeat(data_dict["u_val"].reshape((-1, 1)), 3, axis=1)

    # Designed to make position, acceleration, and potential all exist between [-1,1]
    x_train = x_transformer.fit_transform(data_dict["x_train"])
    u_train = u_transformer.fit_transform(u_train_vals)[:, 0].reshape((-1, 1))

    # Scale the acceleration by the potential
    scaler = 1 / (x_transformer.scale_ / u_transformer.scale_)
    a_bar = a_transformer.fit_transform(
        data_dict["a_train"],
        scaler=scaler,
    )  # this is a_bar
    # a_bar_transformer.fit(a_bar)
    a_train = a_bar  # this should be on the order of [-100, 100]
    # config["a_bar_transformer"] = [a_bar_transformer]

    x_val = x_transformer.transform(data_dict["x_val"])
    a_val = a_transformer.transform(data_dict["a_val"])
    u_val = u_transformer.transform(u_valid_vals)[:, 0].reshape((-1, 1))

    ref_radius_max = config.get("ref_radius_max", config["radius_max"])[0]
    ref_r_vec = np.array([[ref_radius_max, 0, 0]])
    ref_r_vec = x_transformer.transform(ref_r_vec)
    config["ref_radius_max"] = [ref_r_vec[0, 0].astype(np.float32)]

    data_dict = {
        "x_train": x_train,
        "a_train": a_train,
        "u_train": u_train,
        "x_val": x_val,
        "a_val": a_val,
        "u_val": u_val,
    }

    transformers = {
        "x": x_transformer,
        "a": a_transformer,
        "u": u_transformer,
        # "a_bar": a_bar_transformer,
    }
    return data_dict, transformers


def scale_by_non_dimensional_radius(data_dict, config):
    x_transformer = config["x_transformer"][0]
    a_transformer = config["a_transformer"][0]
    u_transformer = config["u_transformer"][0]
    a_bar_transformer = config["a_transformer"][0]

    u_train_vals = np.repeat(data_dict["u_train"], 3, axis=1)
    u_valid_vals = np.repeat(data_dict["u_val"], 3, axis=1)

    # Scale positions by the radius of the planet
    x_train = x_transformer.fit_transform(
        data_dict["x_train"],
        scaler=1 / (config["planet"][0].radius),
    )
    u_train = u_transformer.fit_transform(u_train_vals)[:, 0].reshape((-1, 1))

    # Scale the acceleration by the potential
    scaler = 1 / (x_transformer.scale_ / u_transformer.scale_)
    a_bar = a_transformer.fit_transform(
        data_dict["a_train"],
        scaler=scaler,
    )  # this is a_bar
    a_bar_transformer.fit(a_bar)
    a_train = a_bar  # this should be on the order of [-100, 100]
    config["a_bar_transformer"] = [a_bar_transformer]

    x_val = x_transformer.transform(data_dict["x_val"])
    a_val = a_transformer.transform(data_dict["a_val"])
    u_val = u_transformer.transform(u_valid_vals)[:, 0].reshape((-1, 1))

    data_dict = {
        "x_train": x_train,
        "a_train": a_train,
        "u_train": u_train,
        "x_val": x_val,
        "a_val": a_val,
        "u_val": u_val,
    }

    transformers = {
        "x": x_transformer,
        "a": a_transformer,
        "u": u_transformer,
        "a_bar": a_bar_transformer,
    }
    return data_dict, transformers


def scale_by_constants(data_dict, config):
    x_transformer = config["x_transformer"][0]
    a_transformer = config["a_transformer"][0]
    u_transformer = config["u_transformer"][0]
    a_bar_transformer = config["a_transformer"][0]

    """
    non-dimensionalize by units, not by values
    x = x_tilde / x_star # length
    m = m_tilde / m_star # mass
    t = t_tilde / t_star # time
    """
    x_norm = np.linalg.norm(data_dict["x_train"], axis=1)
    x_star = 10 ** np.mean(np.log10(x_norm))  # average magnitude

    # scale time coordinate based on what makes the accelerations behave nicely
    a_norm = np.linalg.norm(data_dict["a_train"], axis=1)
    a_star = 10 ** np.mean(np.log10(a_norm))  # average magnitude acceleration
    a_star_tmp = a_star / x_star
    t_star = np.sqrt(1 / a_star_tmp)

    x_train = x_transformer.fit_transform(data_dict["x_train"], scaler=1 / x_star)
    a_train = a_transformer.fit_transform(
        data_dict["a_train"],
        scaler=1 / (x_star / t_star**2),
    )
    u_train = u_transformer.fit_transform(
        data_dict["u_train"],
        scaler=1 / (x_star / t_star) ** 2,
    )

    u3vec = np.repeat(data_dict["u_val"], 3, axis=1)

    x_val = x_transformer.transform(data_dict["x_val"])
    a_val = a_transformer.transform(data_dict["a_val"])
    u_val = u_transformer.transform(u3vec)[:, 0].reshape((-1, 1))

    # can't just select max from non-dim x_train because config is dimensionalized
    ref_radius_min = config.get("ref_radius_min", [x_norm.min()])[0]
    ref_radius_max = config.get("ref_radius_max", [x_norm.max()])[0]
    ref_radius_analytic = config.get("ref_radius_analytic", [x_norm.max()])[0]
    x_vec = np.array([[ref_radius_min, ref_radius_max, ref_radius_analytic]])
    x_vec_normalized = x_transformer.transform(x_vec)
    config["ref_radius_min"] = [x_vec_normalized[0, 0]]
    config["ref_radius_max"] = [x_vec_normalized[0, 1]]
    config["ref_radius_analytic"] = [x_vec_normalized[0, 2]]

    if config.get("mu", [None])[0] is not None:
        config["mu_non_dim"] = [config["mu"][0] * (t_star**2) / (x_star) ** 3]

    data_dict = {
        "x_train": x_train,
        "a_train": a_train,
        "u_train": u_train,
        "x_val": x_val,
        "a_val": a_val,
        "u_val": u_val,
    }

    transformers = {
        "x": x_transformer,
        "a": a_transformer,
        "u": u_transformer,
        "a_bar": a_bar_transformer,
    }
    return data_dict, transformers


def scale_by_non_dim_potential(data_dict, config):
    x_transformer = config["x_transformer"][0]
    a_transformer = config["a_transformer"][0]
    u_transformer = config["u_transformer"][0]
    a_bar_transformer = config["a_transformer"][0]

    """
    non-dimensionalize by units, not by values
    x = x_tilde / x_star # length
    m = m_tilde / m_star # mass
    t = t_tilde / t_star # time
    """
    x_norm = np.linalg.norm(data_dict["x_train"], axis=1)
    # x_star = 10**np.mean(np.log10(x_norm)) # average magnitude
    x_star = config["planet"][0].radius

    # scale time coordinate based on what makes the accelerations behave nicely
    u_brill = config["mu"][0] / config["planet"][0].radius

    # Make u_star approximately equal to value of the potential that must be learned.
    # If deg_removed == -1: u_brill
    # If deg_removed == 2: u_J2/u_brill
    if config.get("fuse_models", [False])[0]:
        x = data_dict["x_train"]
        r = np.linalg.norm(x, axis=1)
        u = x[:, 2] / r

        a = config["planet"][0].radius
        mu = config["mu"][0]
        C20 = config.get("cBar", [np.zeros((3, 3))])[0][2, 0]

        u_pm = mu / r

        c1 = np.sqrt(15.0 / 4.0) * np.sqrt(3.0)
        c2 = np.sqrt(5.0 / 4.0)

        u_C20 = (a / r) ** 2 * (mu / r) * (u**2 * c1 - c2) * C20

        u_analytic = -1.0 * (u_pm + u_C20)
        u_sans_J2 = data_dict["u_train"] - u_analytic
        u_max = np.max(np.abs(u_sans_J2))

        if u_max == 0.0:  # this means that the grav model was point mass
            u_max = u_brill * 0.01  # arbitrarily set to 1 percent of point mass

    else:
        u_max = np.max(np.abs(data_dict["u_train"]))
        if u_max == 0.0:  # i.e. not specified
            u_max = u_brill

    u_star = (u_max / u_brill) * u_brill

    t_star = np.sqrt(x_star**2 / u_star)

    x_train = x_transformer.fit_transform(data_dict["x_train"], scaler=1 / x_star)
    a_train = a_transformer.fit_transform(
        data_dict["a_train"],
        scaler=1 / (x_star / t_star**2),
    )
    u_train = u_transformer.fit_transform(
        data_dict["u_train"],
        scaler=1 / u_star,
    )  # u_star == (x_star/t_star)**2

    x_val = x_transformer.transform(data_dict["x_val"])
    a_val = a_transformer.transform(data_dict["a_val"])
    u_val = u_transformer.transform(data_dict["u_val"])

    # can't just select max from non-dim x_train because config is dimensionalized
    ref_radius_min = config.get("ref_radius_min", [x_norm.min()])[0]
    ref_radius_max = config.get("ref_radius_max", [x_norm.max()])[0]
    ref_radius_analytic = config.get("ref_radius_analytic", [x_norm.max()])[0]
    x_vec = np.array([[ref_radius_min, ref_radius_max, ref_radius_analytic]])
    x_vec_normalized = x_transformer.transform(x_vec)
    config["ref_radius_min"] = [x_vec_normalized[0, 0]]
    config["ref_radius_max"] = [x_vec_normalized[0, 1]]
    config["ref_radius_analytic"] = [x_vec_normalized[0, 2]]

    if config.get("mu", [None])[0] is not None:
        config["mu_non_dim"] = [config["mu"][0] * (t_star**2) / (x_star) ** 3]

    data_dict = {
        "x_train": x_train,
        "a_train": a_train,
        "u_train": u_train,
        "x_val": x_val,
        "a_val": a_val,
        "u_val": u_val,
    }

    transformers = {
        "x": x_transformer,
        "a": a_transformer,
        "u": u_transformer,
        "a_bar": a_bar_transformer,
    }
    return data_dict, transformers


def no_scale(data_dict, config):
    dummy_transformer = config.get("dummy_transformer", [DummyScaler()])
    x_transformer = dummy_transformer
    a_transformer = dummy_transformer
    u_transformer = dummy_transformer
    a_bar_transformer = dummy_transformer

    transformers = {
        "x": x_transformer,
        "a": a_transformer,
        "u": u_transformer,
        "a_bar": a_bar_transformer,
    }
    return data_dict, transformers


def single_training_validation_split(X, N_train, N_val, random_state=42):
    """Function responsible for splitting the variable into separate training
    and validation sets"""
    X = shuffle(X, random_state=random_state)
    X_train = X[:N_train]
    X_val = X[N_train : N_train + N_val]
    return X_train, X_val


def training_validation_split(X, Y, Z, N_train, N_val, random_state=42):
    """Function which automates splitting the training and validation data
    for all variables (typically position, acceleration, and potential)"""
    X_train, X_val = single_training_validation_split(
        X,
        N_train,
        N_val,
        random_state=random_state,
    )
    Y_train, Y_val = single_training_validation_split(
        Y,
        N_train,
        N_val,
        random_state=random_state,
    )
    Z_train, Z_val = single_training_validation_split(
        Z,
        N_train,
        N_val,
        random_state=random_state,
    )

    return X_train, Y_train, Z_train, X_val, Y_val, Z_val


def cart2sph_tf(x, acc_N):
    X = x[:, 0]
    Y = x[:, 1]
    Z = x[:, 2]
    r = tf.norm(x, axis=1)
    theta = tf.atan2(Y, X)
    phi = tf.atan2(tf.sqrt(tf.square(X) + tf.square(Y)), Z)
    tf.stack((r, theta, phi), axis=1)

    s_phi = tf.sin(phi)
    c_phi = tf.cos(phi)
    s_theta = tf.sin(theta)
    c_theta = tf.cos(theta)

    r_hat = tf.stack((s_phi * c_theta, s_phi * s_theta, c_phi), axis=1)
    theta_hat = tf.stack((c_phi * c_theta, c_phi * s_theta, -s_phi), axis=1)
    phi_hat = tf.stack((-s_theta, c_theta, tf.zeros_like(s_theta)), axis=1)

    BN = tf.reshape(tf.stack((r_hat, theta_hat, phi_hat), 1), (-1, 3, 3))
    acc_N_3d = tf.reshape(acc_N, (-1, 3, 1))

    # apply BN rotation to each acceleration
    acc_B_3d = tf.einsum("bij,bjk->bik", BN, acc_N_3d)
    acc_B = tf.reshape(acc_B_3d, (-1, 3))

    return acc_B


class DataSet:
    def __init__(self, data_config=None):
        # populate these variables
        self.train_data = None
        self.valid_data = None
        self.transformers = None

        if data_config is not None:
            self.from_config(data_config)
        else:
            self.config = {}

    def get_raw_data(self):
        """Function responsible for getting the raw training data (without
        any preprocessing). This may include concatenating an "extra" training
        data distribution defined within config.

        Args:
            config (dict): hyperparameters and configuration variables for TF Model

        Returns:
            tuple: x,a,u training and validation data
        """
        planet = self.config["planet"][0]
        radius_bounds = [self.config["radius_min"][0], self.config["radius_max"][0]]
        N_dist = self.config["N_dist"][0]

        grav_file = self.config.get("grav_file", [None])[0]

        # HACK: This is a hack to get the correct gravity file for the distribution
        obj_file = (
            grav_file
            if grav_file is not None
            else self.config.get("obj_file", [None])[0]
        )
        sh_file = (
            grav_file
            if grav_file is not None
            else self.config.get("sh_file", [None])[0]
        )
        self.config["obj_file"] = [obj_file]
        self.config["sh_file"] = [sh_file]

        if isinstance(planet, Planet):
            grav_file = sh_file
        else:
            grav_file = obj_file

        distribution = self.config["distribution"][0]
        if distribution.__name__ == "SurfaceDist":
            trajectory = distribution(
                planet,
                make_windows_path_posix(obj_file),
                # **self.config,
            )
        else:
            trajectory = distribution(
                planet,
                radius_bounds,
                N_dist,
                **self.config,
            )
        get_analytic_data_fcn = self.config["gravity_data_fcn"][0]

        x_unscaled, a_unscaled, u_unscaled = get_analytic_data_fcn(
            trajectory,
            grav_file,
            parallel=True,
            **self.config,
        )

        # This condition is for meant to correct for when gravity models didn't always
        # have the proper sign of the potential.
        # TODO: This should be removed prior to production.
        deg_removed = self.config.get("deg_removed", -1)
        remove_point_mass = self.config.get("remove_point_mass", False)
        # if part of the model is removed (i.e. point mass) it is reasonable for some of
        # the potential to be > 0.0, so only rerun if there is no degree removed.
        if np.max(u_unscaled) > 0.0 and deg_removed == -1 and not remove_point_mass:
            sys.exit(
                "ERROR: This pickled acceleration/potential pair was generated \
                when the potential had a wrong sign. \n You must overwrite the data!",
            )

        x_train, a_train, u_train, x_val, a_val, u_val = training_validation_split(
            x_unscaled,
            a_unscaled,
            u_unscaled,
            self.config["N_train"][0],
            self.config["N_val"][0],
            random_state=self.config.get("seed", [42])[0],
        )

        data_dict = {
            "x_train": x_train,
            "a_train": a_train,
            "u_train": u_train,
            "x_val": x_val,
            "a_val": a_val,
            "u_val": u_val,
        }

        if "augment_data_config" in self.config:
            # augment the original data (example: high altitude point mass data)
            augment_config = copy.deepcopy(self.config)
            augment_config.update(augment_config["augment_data_config"][0])
            augment_config.pop("augment_data_config")
            new_dataset = DataSet(augment_config)
            for key in data_dict.keys():
                current_data = data_dict[key]
                new_data = new_dataset.raw_data[key]
                data_dict[key] = np.concatenate([current_data, new_data])

        print_stats(data_dict["x_train"], "Position")
        print_stats(data_dict["a_train"], "Acceleration")
        print_stats(data_dict["u_train"], "Potential")

        return data_dict

    def get_preprocessed_data(self, data_dict):
        """Function responsible for normalizing the training data. Possible options
        include normalizing by the bounds of the acceleration, the potential, neither,
        or in a manner that non-dimensionalizes the equations (this amounts to scaling
        the acceleration in proportion to the more fundamental scalar potential and
        position)."""

        data_dict = add_error(
            data_dict,
            self.config.get("acc_noise", [0.0])[0],
        )

        # Preprocessing
        scale_by = self.config.get("scale_by", ["none"])[0]
        if scale_by == "a":
            preprocess_fcn = scale_by_acceleration
        elif scale_by == "u":
            preprocess_fcn = scale_by_potential
        elif scale_by == "non_dim":
            preprocess_fcn = scale_by_non_dimensional
        elif scale_by == "non_dim_radius":
            preprocess_fcn = scale_by_non_dimensional_radius
        elif scale_by == "non_dim_v2":
            preprocess_fcn = scale_by_constants
        elif scale_by == "non_dim_v3":
            preprocess_fcn = scale_by_non_dim_potential
        elif scale_by == "none":
            preprocess_fcn = no_scale

        data_dict, transformers = preprocess_fcn(data_dict, self.config)

        x_train = data_dict["x_train"]
        a_train = data_dict["a_train"]
        u_train = data_dict["u_train"]
        x_val = data_dict["x_val"]
        a_val = data_dict["a_val"]
        u_val = data_dict["u_val"]

        print_stats(x_train, "Scaled Position")
        print_stats(a_train, "Scaled Acceleration")
        print_stats(u_train, "Scaled Potential")

        laplace_train = np.zeros_like(u_train)
        laplace_val = np.zeros_like(u_val)

        curl_train = np.zeros_like(a_train)
        curl_val = np.zeros_like(a_val)

        train_tuple = (x_train, u_train, a_train, laplace_train, curl_train)
        val_tuple = (x_val, u_val, a_val, laplace_val, curl_val)

        return (train_tuple, val_tuple, transformers)

    def generate_tensorflow_dataset(self, x, y, batch_size, shuffle=True, dtype=None):
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
        if shuffle:
            dataset = dataset.shuffle(len(x), seed=1234)
        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.cache()

        # Why Cache is Impt: https://stackoverflow.com/questions/48240573/why-is-tensorflows-tf-data-dataset-shuffle-so-slow
        return dataset

    def configure_dataset(self, train_data, val_data, config):
        """Function that partitions the training data to include only that
        which is required for the use PINN constraint. I.e. if using the
        AP constraint, there is no need to send extra vectors for L and C
        onto the device (GPU) and slow calculations."""
        x_train, u_train, a_train, laplace_train, curl_train = train_data
        x_val, u_val, a_val, laplace_val, curl_val = val_data
        pinn_constraint_fcn = config.get("PINN_constraint_fcn", ["pinn_00"])[0]

        data = OrderedDict(
            {
                "potential": u_train,
                "acceleration": a_train,
                "laplace": laplace_train,
                "curl": curl_train,
            },
        )
        val_data = OrderedDict(
            {
                "potential": u_val,
                "acceleration": a_val,
                "laplace": laplace_val,
                "curl": curl_val,
            },
        )

        constraint_str = pinn_constraint_fcn.split("_")[1].lower()
        if "a" not in constraint_str and constraint_str != "00":
            data.pop("acceleration")
            val_data.pop("acceleration")
        if "p" not in constraint_str:
            data.pop("potential")
            val_data.pop("potential")
        if "l" not in constraint_str:
            data.pop("laplace")
            val_data.pop("laplace")
        if "c" not in constraint_str:
            data.pop("curl")
            val_data.pop("curl")

        # stack the data into a single array, but
        # ensure that dimensions are 2D

        def hstack_2D(data):
            data_list = []
            for values in data.values():
                if len(np.shape(values)) == 1:
                    values = values.reshape((-1, 1))
                data_list.append(values)
            return np.hstack(data_list)

        y_train = hstack_2D(data)
        y_val = hstack_2D(val_data)

        batch_size = config.get("batch_size", [len(y_train)])[0]
        dtype = config.get("dtype", [tf.float64])[0]
        dataset = self.generate_tensorflow_dataset(
            x_train,
            y_train,
            batch_size,
            dtype=dtype,
        )
        val_dataset = self.generate_tensorflow_dataset(
            x_val,
            y_val,
            batch_size,
            shuffle=False,
            dtype=dtype,
        )

        return dataset, val_dataset

    def add_transformers_to_config(self):
        self.config["x_transformer"] = [self.transformers.get("x", None)]
        self.config["u_transformer"] = [self.transformers.get("u", None)]
        self.config["a_transformer"] = [self.transformers.get("a", None)]
        self.config["a_bar_transformer"] = [self.transformers.get("a_bar", None)]

    def from_config(self, config):
        self.config = config
        data_dict = self.get_raw_data()
        train_data, val_data, transformers = self.get_preprocessed_data(data_dict)
        dataset, val_dataset = self.configure_dataset(train_data, val_data, self.config)

        self.raw_data = data_dict
        self.train_data = dataset
        self.valid_data = val_dataset
        self.transformers = transformers

        # force transformers into config
        self.add_transformers_to_config()

    def from_raw_data(self, x, a, percent_validation=0.1):
        N_train = int(np.round(len(x) * (1.0 - percent_validation)))
        N_val = int(np.round(len(x) * percent_validation))
        x_train, x_val = single_training_validation_split(
            x,
            N_train,
            N_val,
            random_state=42,
        )
        a_train, a_val = single_training_validation_split(
            a,
            N_train,
            N_val,
            random_state=42,
        )
        u_train = np.zeros((N_train, 1))
        u_val = np.zeros((N_val, 1))

        data_dict = {
            "x_train": x_train,
            "a_train": a_train,
            "u_train": u_train,
            "x_val": x_val,
            "a_val": a_val,
            "u_val": u_val,
        }

        train_data, val_data, transformers = self.get_preprocessed_data(data_dict)
        dataset, val_dataset = self.configure_dataset(train_data, val_data, self.config)

        self.raw_data = data_dict
        self.train_data = dataset
        self.valid_data = val_dataset
        self.transformers = transformers

    def add_dataset(self, new_dataset):
        # append new data
        data_dict = self.raw_data.copy()
        for key in data_dict.keys():
            current_data = data_dict[key]
            new_data = new_dataset.raw_data[key]
            data_dict[key] = np.concatenate([current_data, new_data])

        train_data, val_data, transformers = self.get_preprocessed_data(data_dict)
        dataset, val_dataset = self.configure_dataset(train_data, val_data, self.config)

        self.raw_data = data_dict
        self.train_data = dataset
        self.valid_data = val_dataset
        self.transformers = transformers
