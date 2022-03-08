import tensorflow as tf

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




def weighted_percent_rms_loss(rms_components, percent_error):
    loss = tf.reduce_sum(tf.reduce_sum(rms_components,1)*percent_error)#*tf.reduce_sum(percent_error)
    return loss

