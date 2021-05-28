
import tensorflow as tf


def update_constant(tape, loss_components, constant_avg, beta, variables):
    # Get max residual weight
    gradient = tape.gradient(loss_components[0], variables,  unconnected_gradients='zero')
    max_grad_res_list = []
    for grad in gradient:
        max_grad_res_list.append(tf.reduce_max(tf.abs(grad)))
    max_grad_res = tf.reduce_max(tf.stack(max_grad_res_list))

    # Get the mean boundary condition weights
    mean_grad_bc_list = []
    adaptive_constants = []
    for i in range(0, loss_components.shape[0]):
        gradient = tape.gradient(loss_components[i], variables,  unconnected_gradients='zero')
        for grad in gradient:
            mean_grad_bc_list.append(tf.reduce_max(tf.abs(grad)))
        mean_grad_bc = tf.reduce_mean(tf.abs(tf.stack(mean_grad_bc_list)))
        adaptive_constants.append(max_grad_res / mean_grad_bc)
        
    # generate new adaptive constants
    adaptive_constants = tf.stack(adaptive_constants)

    # update adaptive constant (rolling average)
    constant_avg = constant_avg*tf.subtract(1.0,beta) + beta*adaptive_constants
    return constant_avg

def no_pinn_anneal(loss_components, adaptive_const):
    loss_res = loss_components
    return loss_res

def pinn_A_anneal(loss_components, adaptive_const):
    loss_res = loss_components
    return loss_res

def pinn_AP_anneal(loss_components, adaptive_const):
    loss_res = loss_components[3]
    loss_A = loss_components[0:3]*adaptive_const[1]
    return (loss_res, loss_A)

def pinn_AL_anneal(loss_components, adaptive_const):
    loss_res = loss_components[0:3]
    loss_bc = loss_components[3:]*adaptive_const[1]
    return (loss_res, loss_bc)

def pinn_APL_anneal(loss_components, adaptive_const):
    loss_res = loss_components[0:3]
    loss_P = loss_components[3]*adaptive_const[1]
    loss_L = loss_components[4]*adaptive_const[2]
    return (loss_res, loss_P, loss_L)

def pinn_ALC_anneal(loss_components, adaptive_const):
    loss_res = loss_components[0:3]
    loss_L = loss_components[3]*adaptive_const[1]
    loss_C = loss_components[4:7]*adaptive_const[2]
    return (loss_res, loss_L, loss_C)

def pinn_APLC_anneal(loss_components, adaptive_const):
    loss_res = loss_components[3]
    loss_A = loss_components[0:3]*adaptive_const[1]
    loss_L = loss_components[4]*adaptive_const[2]
    loss_C = loss_components[5:8]*adaptive_const[3]
    return (loss_res, loss_A, loss_L, loss_C)


