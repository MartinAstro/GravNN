from collections import OrderedDict
import tensorflow as tf

def get_annealing_fcn(use_anneal):
    from GravNN.Networks.Annealing import update_w_loss, hold_constant
    if use_anneal:
        return update_w_loss
    else:
        return hold_constant

def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator/denominator

def hold_constant(w_loss, train_counter, losses, variables, tape):
    return w_loss

# ANNEALING V2
def update_w_loss(w_loss, train_counter, losses, variables, tape):
    traces = []
    update_interval = tf.constant(1000, dtype=tf.int64)
    min_start_idx = tf.constant(100, dtype=tf.int64)
    N_samples = tf.constant(50, dtype=tf.int64)

    if tf.math.mod(train_counter, update_interval) == 0 and \
        train_counter > min_start_idx:
        for loss_i in losses.values():
            jacobian = tape.jacobian(loss_i, variables) 
            gradients = []
            for i in range(len(jacobian)-1): #batch size
                gradients.append(tf.reshape(jacobian[i], (len(jacobian),-1))) # flatten
            J = tf.concat(gradients, 1)
            K_i = J@tf.transpose(J) # NTK of the values  [N_samples x N_samples]
            trace_K_i = tf.linalg.trace(K_i)
            traces.append(trace_K_i)
        trace_K = tf.reduce_sum(traces)
        w_loss_new = tf.stack([trace_K/trace for trace in traces],0)
        w_loss.assign(w_loss_new)
        tf.print(w_loss)
    return w_loss
