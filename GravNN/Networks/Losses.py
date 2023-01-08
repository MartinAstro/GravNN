import tensorflow as tf
import numpy as np

def get_loss_fcn(loss_key):
    return { 
        "rms" : rms,
        "percent" : percent,
        "angle" : angle,
        "magnitude" : magnitude

    }[loss_key.lower()]


def MetaLoss(y_hat, y, loss_fcn_list):
    losses = {}
    for loss_fcn in loss_fcn_list:
        losses.update({loss_fcn.__name__ : loss_fcn(y_hat, y)})
    return losses


def rms(y_hat, y):
    dy = y_hat - y
    return tf.sqrt(tf.reduce_sum(tf.square(dy), axis=1))

def percent(y_hat, y):
    #https://github.com/tensorflow/tensorflow/issues/12071
    da = tf.subtract(y_hat[:,0:3], y[:,0:3]) + tf.constant(1.0e-16, dtype=y.dtype) # if perfectly zero, nan's
    da_norm = tf.norm(da, axis=1)
    a_true_norm = tf.norm(y[:,0:3],axis=1)
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