import tensorflow as tf


def compute_rms_components(y_hat, y):
    """Separate the different loss component terms.

    Args:
        y_hat (tf.Tensor): predicted values
        y (tf.Tensor): true values

    Returns:
        tf.Tensor: loss components for each contribution (i.e. dU, da, dL, dC)
    """
    loss_components = tf.square(tf.subtract(y_hat, y))
    return loss_components

def compute_percent_error(y_hat, y):
    # Only apply to the acceleration values 
    da = tf.subtract(y_hat[:,0:3], y[:,0:3])
    da_norm = tf.norm(da, axis=1)
    a_true_norm = tf.norm(tf.abs(y[:,0:3]),axis=1)
    percent_multiplier = tf.constant(100, dtype=da.dtype)
    loss_components = da_norm/a_true_norm*percent_multiplier
    return loss_components



# Max Losses
def max_loss(rms_components, percent_error): 
    percent_loss = percent_summed_loss(rms_components, percent_error)
    max_loss = tf.reduce_max(percent_error) 
    return max_loss


# Summed Losses
def percent_summed_loss(rms_components, percent_error):
    loss = tf.reduce_sum(percent_error)
    return loss

def rms_summed_loss(rms_components, percent_error): 
    loss = tf.reduce_sum(tf.sqrt(tf.reduce_sum(rms_components, 1)))
    return loss

def percent_rms_summed_loss(rms_components, percent_error):
    rms_loss = rms_summed_loss(rms_components, percent_error)
    percent_loss = percent_summed_loss(rms_components, percent_error)
    loss = rms_loss + percent_loss
    return loss


# Average Losses -- Effectively the same as ''Summed Losses'' except 1/N
def percent_avg_loss(rms_components, percent_error):
    loss = tf.reduce_mean(percent_error)
    return loss

def rms_avg_loss(rms_components, percent_error):
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(rms_components, 1)))
    return loss

def percent_rms_avg_loss(rms_components, percent_error):
    rms_loss = rms_avg_loss(rms_components, percent_error)
    percent_loss = percent_avg_loss(rms_components, percent_error)
    loss = rms_loss + percent_loss
    return loss


# Hybrid Losses (mix of summed, averaged, and maximum losses)
def avg_percent_summed_rms_loss(rms_components, percent_error):
    rms_loss = rms_summed_loss(rms_components, percent_error)
    percent_loss = percent_avg_loss(rms_components, percent_error)
    loss = rms_loss + percent_loss
    return loss

def avg_percent_summed_rms_max_error_loss(rms_components, percent_error):
    sum_rms = rms_summed_loss(rms_components, percent_error)
    avg_percent = percent_avg_loss(rms_components, percent_error)
    max_percent = tf.reduce_max(percent_error) 
    loss = sum_rms + avg_percent + max_percent
    return loss

def weighted_mean_percent_loss(rms_components, percent_error):
    w_i = tf.reduce_sum(rms_components,1)
    w_sum = tf.reduce_sum(w_i)
    L_i = percent_error
    loss = tf.reduce_sum(tf.multiply(w_i,L_i)) / w_sum
    return loss

