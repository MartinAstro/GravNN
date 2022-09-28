"""Different Physics Informed constraints that can be used to train the network. """

import tensorflow as tf

# signature needed to load a prior ALC network
@tf.function(input_signature=[tf.TensorSpec(shape=(None,3,3), dtype=tf.float32)])
def laplacian(u_xx):
    return tf.reduce_sum(tf.linalg.diag_part(u_xx), 1, keepdims=True)

@tf.function(input_signature=[tf.TensorSpec(shape=(None,3,3), dtype=tf.float64)])
def laplacian(u_xx):
    return tf.reduce_sum(tf.linalg.diag_part(u_xx), 1, keepdims=True)


def no_pinn(f, x, training):
    u_x = f(x, training)
    return u_x


def pinn_A(f, x, training):
    with tf.GradientTape() as tape:
        tape.watch(x)
        u = f(x, training)
    a_x = -tape.gradient(u, x)
    return a_x

def pinn_A_Ur(f, x, training):
    with tf.GradientTape() as tape:
        tape.watch(x)
        ur = f(x, training)
    ur_x = tape.gradient(ur, x)
    r = tf.norm(x, axis=1)
    a_x = x[:,0]/r**3*ur[:,0]/r - ur_x[:,0]/r
    a_y = x[:,1]/r**3*ur[:,0]/r - ur_x[:,1]/r
    a_z = x[:,2]/r**3*ur[:,0]/r - ur_x[:,2]/r
    a = tf.stack((a_x,a_y, a_z),axis=1)
    return a

def pinn_P(f, x, training):
    u = f(x, training)
    return u


def pinn_PL(f, x, training):
    with tf.GradientTape(persistent=False) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training)  # shape = (k,) #! evaluate network
        u_x = g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative

    # https://github.com/tensorflow/tensorflow/issues/40885 -- batch_jacobian doesn't work with experimental compile
    # It does in 2.5, but there is a known issue for the implimentation: https://www.tensorflow.org/xla/known_issues?hl=nb
    # The while_loop needs to be bounded
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)
    laplace = laplacian(u_xx)
    return tf.concat((u, laplace), 1)


def pinn_PLC(f, x, training):
    with tf.GradientTape(persistent=False) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training)  # shape = (k,) #! evaluate network
        u_x = g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative

    # https://github.com/tensorflow/tensorflow/issues/40885 -- batch_jacobian doesn't work with experimental compile
    # It does in 2.5, but there is a known issue for the implimentation: https://www.tensorflow.org/xla/known_issues?hl=nb
    # The while_loop needs to be bounded
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)
    laplace = laplacian(u_xx)

    curl_x = tf.math.subtract(u_xx[:, 2, 1], u_xx[:, 1, 2])
    curl_y = tf.math.subtract(u_xx[:, 0, 2], u_xx[:, 2, 0])
    curl_z = tf.math.subtract(u_xx[:, 1, 0], u_xx[:, 0, 1])

    curl = tf.stack([curl_x, curl_y, curl_z], axis=1)
    return tf.concat((u, laplace, curl), 1)


def pinn_AP(f, x, training):
    with tf.GradientTape() as tape:
        tape.watch(x)
        u = f(x, training)
    u_x = tape.gradient(u, x)
    accel = tf.multiply(u_x, -1.0) # u_x must be first s.t. -1 dtype is inferred
    return tf.concat((u, accel), 1)


def pinn_AL(f, x, training):
    with tf.GradientTape(persistent=False) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training)  # shape = (k,) #! evaluate network
        u_x = g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative

    accel = tf.multiply(u_x, -1.0) # u_x must be first s.t. -1 dtype is inferred

    # https://github.com/tensorflow/tensorflow/issues/40885 -- batch_jacobian doesn't work with experimental compile
    # It does in 2.5, but there is a known issue for the implimentation: https://www.tensorflow.org/xla/known_issues?hl=nb
    # The while_loop needs to be bounded
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)
    laplace = laplacian(u_xx)
    return tf.concat((accel, laplace), 1)


def pinn_ALC(f, x, training):
    with tf.GradientTape(persistent=False) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training)  # shape = (k,) #! evaluate network
        u_x = g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)

    accel = tf.multiply(u_x, -1.0) # u_x must be first s.t. -1 dtype is inferred

    laplace = laplacian(u_xx)

    curl_x = tf.math.subtract(u_xx[:, 2, 1], u_xx[:, 1, 2])
    curl_y = tf.math.subtract(u_xx[:, 0, 2], u_xx[:, 2, 0])
    curl_z = tf.math.subtract(u_xx[:, 1, 0], u_xx[:, 0, 1])

    curl = tf.stack([curl_x, curl_y, curl_z], axis=1)
    return tf.concat((accel, laplace, curl), 1)


def pinn_APL(f, x, training):
    with tf.GradientTape(persistent=False) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training)  # shape = (k,) #! evaluate network
        u_x = g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative

    accel = tf.multiply(u_x, -1.0) # u_x must be first s.t. -1 dtype is inferred

    # https://github.com/tensorflow/tensorflow/issues/40885 -- batch_jacobian doesn't work with experimental compile
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)
    laplace = laplacian(u_xx)
    return tf.concat((u, accel, laplace), 1)


def pinn_APLC(f, x, training):
    with tf.GradientTape(persistent=False) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training)  # shape = (k,) #! evaluate network
        u_x = g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)

    accel = tf.multiply(u_x, -1.0) # u_x must be first s.t. -1 dtype is inferred

    laplace = laplacian(u_xx)

    curl_x = tf.math.subtract(u_xx[:, 2, 1], u_xx[:, 1, 2])
    curl_y = tf.math.subtract(u_xx[:, 0, 2], u_xx[:, 2, 0])
    curl_z = tf.math.subtract(u_xx[:, 1, 0], u_xx[:, 0, 1])

    curl = tf.stack([curl_x, curl_y, curl_z], axis=1)

    return tf.concat((u, accel, laplace, curl), 1)
