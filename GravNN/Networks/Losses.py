from collections import OrderedDict

import numpy as np
import tensorflow as tf


def get_loss_fcn(loss_key):
    return {
        "mse": mse,
        "rms": rms,
        "percent": percent,
        "angle": angle,
        "magnitude": magnitude,
    }[loss_key.lower()]


# https://github.com/tensorflow/tensorflow/issues/12071#issuecomment-420279641
@tf.custom_gradient
def norm(x):
    y = tf.norm(x, axis=1, keepdims=True)

    def grad(dy):
        # dy has inf's sometimes
        result = tf.math.multiply_no_nan(
            dy,
            (x / (y + tf.constant(1e-16, dtype=y.dtype))),
        )
        return result

    return y, grad


def MetaLoss(y_hat_dict, y_dict, loss_fcn_list):
    losses = OrderedDict()
    for loss_fcn in loss_fcn_list:
        for key in y_hat_dict.keys() & y_dict.keys():
            # Don't compute percent of laplacian, curl
            loss_name = loss_fcn.__name__
            laplacian_or_curl = key == "laplacian" or key == "curl"
            invalid_percent = loss_name == "percent" and laplacian_or_curl

            # When using ALC loss, only train with percent error on RMS + RMS on L and C
            # biased_rms = loss_name == "rms" and (key == "acceleration")
            if invalid_percent:  #  or biased_rms:
                continue

            y_hat = y_hat_dict[key]
            y = y_dict[key]
            loss = loss_fcn(y_hat, y)

            # Don't hold losses of zero
            if tf.math.count_nonzero(loss) != 0:
                loss_name = f"{key}_{loss_fcn.__name__}"
                losses.update({loss_name: loss})

    return losses


def mse(y_hat, y):
    dy = y_hat - y
    dy_sq = tf.square(dy)
    mse = tf.reduce_mean(dy_sq, axis=1)
    return mse


def rms(y_hat, y):
    dy = y_hat - y
    rms = norm(dy)
    return rms


def percent(y_hat, y):
    da = tf.subtract(y_hat[:, 0:3], y[:, 0:3])
    da_norm = norm(da)
    a_true_norm = norm(y[:, 0:3])
    loss_components = tf.math.divide_no_nan(da_norm, a_true_norm)
    return loss_components


def angle(y_hat, y):
    a_hat = y_hat[:, 0:3]
    a_hat_mag = tf.norm(a_hat, axis=1)

    a = y[:, 0:3]
    a_mag = tf.norm(a, axis=1)

    eps = tf.constant(1e-7, dtype=y.dtype)
    cos_theta = tf.reduce_sum(a_hat * a, axis=1) / (a_hat_mag * a_mag)
    cos_theta_clipped = tf.clip_by_value(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = tf.acos(cos_theta_clipped) / tf.constant(np.pi, dtype=y.dtype)
    return theta


def magnitude(y_hat, y):
    a_hat_mag = tf.norm(y_hat[:, 0:3], axis=1)
    a_mag = tf.norm(y[:, 0:3], axis=1)
    mag_error = tf.abs(a_hat_mag - a_mag) / a_mag
    return mag_error
