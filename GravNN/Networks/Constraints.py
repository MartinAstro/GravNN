"""Different Physics Informed constraints that can be used to train the network. """

import tensorflow as tf


def get_PI_constraint(value):
    """Method responsible for getting all variables / methods used in the physics informed constraint.

    Args:
        value (str): PINN constraint name (i.e. 'pinn_A', 'pinn_aplc', etc)

    Returns:
        list: PINN constraint function, PINN lr annealing function, PINN lr annealing initial values
    """
    from GravNN.Networks.Constraints import (
        pinn_00,
        pinn_A,
        pinn_P,
        pinn_AP,
        pinn_ALC,
        pinn_APLC,
    )

    # Backwards compatibility (if the value is a function -- take the name of the function then select corresponding values)
    try:
        value = value.__name__
    except:
        pass

    # -1 values in the PINN lr annealing initial values indicates that the value will not get updated.
    return {
        "pinn_00": pinn_00,
        "pinn_a": pinn_A,
        "pinn_p": pinn_P,
        "pinn_pl": pinn_P,
        "pinn_ap": pinn_AP,
        "pinn_alc": pinn_ALC,
        "pinn_aplc": pinn_APLC,
    }[value.lower()]


# signature needed to load a prior ALC network
@tf.function(input_signature=[tf.TensorSpec(shape=(None,3,3), dtype=tf.float32)])
def laplacian(u_xx):
    return tf.reduce_sum(tf.linalg.diag_part(u_xx), 1, keepdims=True)

# @tf.function(input_signature=[tf.TensorSpec(shape=(None,3,3), dtype=tf.float64)])
# def laplacian(u_xx):
#     return tf.reduce_sum(tf.linalg.diag_part(u_xx), 1, keepdims=True)


def pinn_00(f, x, training):
    u_x = f(x, training=training)
    return u_x

def pinn_P(f, x, training):
    u = f(x, training=training)
    return u

def pinn_A(f, x, training):
    with tf.GradientTape() as tape:
        tape.watch(x)
        u = f(x, training=training)
    a_x = -tape.gradient(u, x)
    return a_x

def pinn_AP(f, x, training):
    with tf.GradientTape() as tape:
        tape.watch(x)
        u = f(x, training=training)
    u_x = tape.gradient(u, x)
    accel = tf.multiply(u_x, -1.0) # u_x must be first s.t. -1 dtype is inferred
    return tf.concat((u, accel), 1)

def pinn_ALC(f, x, training):
    with tf.GradientTape(persistent=False) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training=training)  # shape = (k,) #! evaluate network
        u_x = g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)

    accel = tf.multiply(u_x, -1.0) # u_x must be first s.t. -1 dtype is inferred

    laplace = laplacian(u_xx)

    curl_x = tf.math.subtract(u_xx[:, 2, 1], u_xx[:, 1, 2])
    curl_y = tf.math.subtract(u_xx[:, 0, 2], u_xx[:, 2, 0])
    curl_z = tf.math.subtract(u_xx[:, 1, 0], u_xx[:, 0, 1])

    curl = tf.stack([curl_x, curl_y, curl_z], axis=1)
    return tf.concat((accel, laplace, curl), 1)

def pinn_APLC(f, x, training):
    with tf.GradientTape(persistent=False) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training=training)  # shape = (k,) #! evaluate network
        u_x = g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)

    accel = tf.multiply(u_x, -1.0) # u_x must be first s.t. -1 dtype is inferred

    laplace = laplacian(u_xx)

    curl_x = tf.math.subtract(u_xx[:, 2, 1], u_xx[:, 1, 2])
    curl_y = tf.math.subtract(u_xx[:, 0, 2], u_xx[:, 2, 0])
    curl_z = tf.math.subtract(u_xx[:, 1, 0], u_xx[:, 0, 1])

    curl = tf.stack([curl_x, curl_y, curl_z], axis=1)

    return tf.concat((u, accel, laplace, curl), 1)



def general_constraint(f, x, training):
    with tf.GradientTape(persistent=False) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training=training)  # shape = (k,) #! evaluate network
        u_x = g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)
    accel = tf.multiply(u_x, -1.0) # u_x must be first s.t. -1 dtype is inferred

    laplace = laplacian(u_xx)

    curl_x = tf.math.subtract(u_xx[:, 2, 1], u_xx[:, 1, 2])
    curl_y = tf.math.subtract(u_xx[:, 0, 2], u_xx[:, 2, 0])
    curl_z = tf.math.subtract(u_xx[:, 1, 0], u_xx[:, 0, 1])

    curl = tf.stack([curl_x, curl_y, curl_z], axis=1)

    outputs = {
        "acceleration" : accel,
        "laplacian" : laplace,
        "curl" : curl
    }
    return outputs
