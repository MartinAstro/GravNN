from collections import OrderedDict

import tensorflow as tf

from GravNN.Networks.Losses import MetaLoss


def get_annealing_fcn(use_anneal):
    from GravNN.Networks.Annealing import hold_constant, update_w_loss

    if use_anneal:
        return update_w_loss
    else:
        return hold_constant


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def compute_loss_subset(y_hat_dict, y_dict, loss_fcn_list):
    batch_size = tf.constant(50, dtype=tf.int64)
    losses_subset = OrderedDict()

    y_hat_dict_subset = OrderedDict()
    for key, values in y_hat_dict.items():
        y_hat_dict_subset[key] = values[:batch_size]

    y_dict_subset = OrderedDict()
    for key, values in y_dict.items():
        y_dict_subset[key] = values[:batch_size]

    losses_subset = MetaLoss(y_hat_dict_subset, y_dict_subset, loss_fcn_list)
    return losses_subset


def hold_constant(w_loss, train_counter, losses, variables, tape):
    return w_loss


# ANNEALING V2
def update_w_loss(w_loss, train_counter, losses, variables, tape):
    traces = []
    update_interval = tf.constant(100, dtype=tf.int64)
    min_start_idx = tf.constant(10, dtype=tf.int64)

    if (
        tf.math.mod(train_counter, update_interval) == 0
        and train_counter > min_start_idx
    ):
        for loss_i in losses.values():
            jacobian = tape.jacobian(loss_i, variables)
            gradients = []
            for i in range(len(jacobian) - 1):  # batch size
                gradients.append(
                    tf.reshape(jacobian[i], (len(jacobian[i]), -1)),
                )  # flatten
            J = tf.concat(gradients, 1)
            K_i = J @ tf.transpose(J)  # NTK of the values  [N_samples x N_samples]
            trace_K_i = tf.linalg.trace(K_i)
            traces.append(trace_K_i)
        trace_K = tf.reduce_sum(traces)
        w_loss_new = tf.stack([trace_K / trace for trace in traces], 0)
        w_loss.assign(w_loss_new)
        tf.print(w_loss)
    return w_loss
