"""Algorithms and support functions to allow for PINN learning rate annealing presented in Wang2020 (
https://www.amcs.upenn.edu/sites/default/files/Understanding%20and%20mitigating%20gradient%20pathologies%20in%20physics-informed%20neural%20networks.pdf
)
"""

import tensorflow as tf


def get_PI_annealing(value):
    """Method responsible for getting all variables / methods used in the physics informed constraint.

    Args:
        value (str): PINN constraint name (i.e. 'pinn_A', 'pinn_aplc', etc)

    Returns:
        list: PINN constraint function, PINN lr annealing function, PINN lr annealing initial values
    """
    from GravNN.Networks.Annealing import (
        pinn_00_anneal,
        pinn_A_anneal,
        pinn_P_anneal,
        pinn_AP_anneal,
        pinn_ALC_anneal,
        pinn_APLC_anneal,
    )

    # Backwards compatibility (if the value is a function -- take the name of the function then select corresponding values)
    try:
        value = value.__name__
    except:
        pass

    # -1 values in the PINN lr annealing initial values indicates that the value will not get updated.
    return {
        "pinn_00": pinn_00_anneal,
        "pinn_a": pinn_A_anneal,
        "pinn_p": pinn_P_anneal,
        "pinn_ap": pinn_AP_anneal,
        "pinn_alc": pinn_ALC_anneal,
        "pinn_aplc": pinn_APLC_anneal,
    }[value.lower()]

def get_PI_adaptive_constants(value):
    # Backwards compatibility (if the value is a function -- take the name of the function then select corresponding values)
    try:
        value = value.__name__
    except:
        pass

    # -1 values in the PINN lr annealing initial values indicates that the value will not get updated.
    return {
        "pinn_00": [1.0], 
        "pinn_a": [1.0], 
        "pinn_p": [1.0],
        "pinn_ap": [1.0, 1.0],
        "pinn_alc": [1.0, 1.0, 1.0],
        "pinn_aplc": [1.0, 1.0, 1.0, 1.0],
    }[value.lower()]

def get_annealing_fcn(name):
    """Helper function to determine if the annealing learning rates of Wang2020
    are going to be used

    Args:
        name (str): key specifying how lr will be annealed

    Returns:
        function: lr annealing method
    """
    from GravNN.Networks.Annealing import update_constant, hold_constant, custom_constant
    return {
        "anneal": update_constant,
        "hold": hold_constant,
        "custom": custom_constant,
    }[name.lower()]


def log10(x):
    numerator = tf.math.log(x)
    denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
    return numerator/denominator

def update_constant(tape, loss_components, constant_avg, beta, variables):
    # Get max residual weight
    gradient = tape.gradient(
        loss_components[0], variables, unconnected_gradients="zero"
    )
    max_grad_res_list = []
    for grad in gradient:
        max_grad_res_list.append(tf.reduce_max(tf.abs(grad)))
    max_grad_res = tf.reduce_max(tf.stack(max_grad_res_list))

    # Get the mean boundary condition weights
    mean_grad_bc_list = []
    adaptive_constants = []
    for loss_component in loss_components:
        gradient = tape.gradient(
            loss_component, variables, unconnected_gradients="zero"
        )
        for grad in gradient:
            mean_grad_bc_list.append(tf.reduce_max(tf.abs(grad)))
        mean_grad_bc = tf.reduce_mean(tf.abs(tf.stack(mean_grad_bc_list)))
        adaptive_constants.append(max_grad_res / mean_grad_bc)

    # generate new adaptive constants
    adaptive_constants = tf.stack(adaptive_constants)
    exponent = tf.constant([0, 1, 1], dtype=beta.dtype)
    adaptive_constants = tf.pow(adaptive_constants, exponent) #forces first component to 1

    # update adaptive constant (rolling average)
    one = tf.constant(1.0,dtype=beta.dtype)
    new_const_avg = constant_avg * tf.subtract(one, beta) + beta * adaptive_constants
    return new_const_avg

def hold_constant(tape, loss_components, constant_avg, beta, variables):
    return constant_avg

def custom_constant(tape, loss_components, constant_avg, beta, variables):
    # loss_log = log10(loss_components[0])
    one = tf.constant(1.0, dtype=beta.dtype)
    new_constants = tf.divide(loss_components[0], loss_components)
    new_adaptive_const = constant_avg * tf.subtract(one, beta) + beta * new_constants
    return new_adaptive_const

def pinn_00_anneal(loss_components, adaptive_const):
    loss_res = loss_components
    return (loss_res,)


def pinn_A_anneal(loss_components, adaptive_const):
    loss_res = loss_components
    return (loss_res,)


def pinn_P_anneal(loss_components, adaptive_const):
    loss_res = loss_components
    return (loss_res,)


def pinn_AP_anneal(loss_components, adaptive_const):
    loss_res = loss_components[:,0] * adaptive_const[0]
    loss_A = tf.reduce_sum(loss_components[:,1:4], axis=1) * adaptive_const[1]
    return (loss_res, loss_A)


def pinn_ALC_anneal(loss_components, adaptive_const):
    loss_res = tf.reduce_sum(loss_components[:,0:3], axis=1)*adaptive_const[0]
    loss_L = loss_components[:,3] * adaptive_const[1]
    loss_C = tf.reduce_sum(loss_components[:,4:7], axis=1) * adaptive_const[2]
    return (loss_res, loss_L, loss_C)


def pinn_APLC_anneal(loss_components, adaptive_const):
    loss_res = loss_components[:,0] * adaptive_const[0]
    loss_A = tf.reduce_sum(loss_components[:,1:4], axis=1) * adaptive_const[1]
    loss_L = loss_components[:,4] * adaptive_const[2]
    loss_C = tf.reduce_sum(loss_components[:,5:8], axis=1) * adaptive_const[3]
    return (loss_res, loss_A, loss_L, loss_C)


# ANNEALING V2
def update_w_loss(w_loss, train_counter, losses, variables, tape):
    traces = []
    if tf.math.mod(train_counter, tf.constant(100, dtype=tf.int64)) == 0:

        for loss_i in losses.values():
            # TODO: This non-deterministically takes up inf RAM. 
            jacobian = tape.jacobian(loss_i, variables)

            gradients = []
            for i in range(len(jacobian)-1): #batch size
                gradients.append(tf.reshape(jacobian[i], (len(jacobian[i]),-1))) # flatten

            J = tf.concat(gradients, 1)

            K_i = J@tf.transpose(J) # NTK of the values  [N_samples x N_samples]
            trace_K_i = tf.linalg.trace(K_i)
            traces.append(trace_K_i)
        trace_K = tf.reduce_sum(traces)
        w_loss = tf.stack([trace_K/trace for trace in traces],0)
        tf.print(w_loss)
    return w_loss
