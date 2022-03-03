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

def percent_rms_loss_v2(rms_components, percent_components):
    rms_loss = tf.reduce_sum(rms_components,1) 
    loss = tf.reduce_sum(rms_loss + percent_components)
    return loss

def weighted_percent_rms_loss(rms_components, percent_components):
    loss = tf.reduce_sum(tf.reduce_sum(rms_components,1)*percent_components)#*tf.reduce_sum(percent_components)
    return loss

def mean_percent_rms_loss(rms_components, percent_components):
    rms_loss = tf.reduce_sum(rms_components,1) 
    loss = tf.reduce_mean(rms_loss + percent_components)
    return loss


def mean_percent_rms_max_loss(rms_components, percent_components):
    rms_loss = tf.math.sqrt(tf.reduce_sum(rms_components,1))
    avg_rms = tf.reduce_mean(rms_loss)
    avg_percent = tf.reduce_mean(percent_components)
    max_percent = tf.reduce_max(percent_components)
    loss = avg_rms + avg_percent + max_percent
    return loss

def mean_percent_rms_max_loss_v2(rms_components, percent_components):
    rms_loss = tf.math.sqrt(tf.reduce_sum(rms_components,1))
    avg_rms = tf.reduce_sum(rms_loss) # sum all the rms
    # print(avg_rms)
    avg_percent = tf.reduce_mean(percent_components) # take the average percent error
    max_percent = tf.reduce_max(percent_components) # penalize very large percent error
    loss = avg_rms + avg_percent + max_percent
    return loss

def mean_percent_rms_loss(rms_components, percent_components):
    rms_loss = tf.math.sqrt(tf.reduce_sum(rms_components,1))
    avg_rms = tf.reduce_sum(rms_loss) # sum all the rms
    # print(avg_rms)
    avg_percent = tf.reduce_mean(percent_components) # take the average percent error
    max_percent = tf.reduce_max(percent_components) # penalize very large percent error
    loss = avg_rms + avg_percent
    return loss