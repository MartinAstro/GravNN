import tensorflow as tf

def percent_loss(rms_components, percent_components):
    loss = tf.reduce_sum(percent_components)
    return loss

def rms_loss(rms_components, percent_components):
    loss = tf.reduce_sum(rms_components)
    return loss

def percent_rms_loss(rms_components, percent_components):
    loss = tf.reduce_sum(rms_components)
    loss += tf.reduce_sum(percent_components)
    return loss