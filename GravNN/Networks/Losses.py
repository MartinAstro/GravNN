import tensorflow as tf
import numpy as np
from collections import OrderedDict
def get_loss_fcn(loss_key):
    return { 
        "rms" : rms,
        "percent" : percent,
        "angle" : angle,
        "magnitude" : magnitude

    }[loss_key.lower()]

#https://github.com/tensorflow/tensorflow/issues/12071#issuecomment-420279641
@tf.custom_gradient
def norm(x):
    y = tf.norm(x, axis=1, keepdims=True)
    def grad(dy):
        #dy has inf's sometimes
        result = tf.math.multiply_no_nan(dy, (x / (y + tf.constant(1e-16, dtype=y.dtype))))
        return result
    return y, grad


def MetaLoss(y_hat_dict, y_dict, loss_fcn_list):
    losses = OrderedDict()
    for loss_fcn in loss_fcn_list:
        for key in y_hat_dict.keys() & y_dict.keys():
            
            # Don't compute percent of laplacian, curl
            if loss_fcn.__name__ == 'percent' and (key == 'laplacian' or key == 'curl'):
                continue

            y_hat = y_hat_dict[key]
            y = y_dict[key]
            loss = loss_fcn(y_hat, y)

            # Don't hold losses of zero
            if tf.math.count_nonzero(loss) != 0:
                loss_name = f"{key}_{loss_fcn.__name__}" 
                losses.update({loss_name : loss}) 

    return losses


def rms(y_hat, y):
    dy = y_hat - y
    return tf.sqrt(norm(dy))

def percent(y_hat, y):
    da = tf.subtract(y_hat[:,0:3], y[:,0:3])
    da_norm = norm(da)
    a_true_norm = norm(y[:,0:3])
    loss_components = tf.math.divide_no_nan(da_norm,a_true_norm) 
    return loss_components


def angle(y_hat, y):
    a_hat = y_hat[:,0:3]
    a_hat_mag = tf.norm(a_hat,axis=1)

    a = y[:,0:3]
    a_mag = tf.norm(a,axis=1)
    
    eps = tf.constant(1E-7, dtype=y.dtype)
    cos_theta = tf.reduce_sum(a_hat*a,axis=1)/(a_hat_mag*a_mag)
    cos_theta_clipped = tf.clip_by_value(cos_theta, -1.0+eps, 1.0-eps)
    theta = tf.acos(cos_theta_clipped) / tf.constant(np.pi, dtype=y.dtype)
    return theta

def magnitude(y_hat, y):
    a_hat_mag = tf.norm(y_hat[:,0:3],axis=1)
    a_mag = tf.norm(y[:,0:3],axis=1)
    mag_error = tf.abs(a_hat_mag - a_mag) / a_mag
    return mag_error