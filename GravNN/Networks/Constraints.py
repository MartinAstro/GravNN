import tensorflow as tf

def no_pinn(f, x, training):
    u_x = f(x, training)
    return u_x

def pinn_A(f, x, training):
    with tf.GradientTape() as tape:
        tape.watch(x)
        u = f(x, training)
    u_x = tape.gradient(u, x)
    return tf.multiply(-1.0,u_x)

def pinn_AP(f, x, training):
    with tf.GradientTape() as tape:
        tape.watch(x)
        u = f(x, training)
    u_x = tape.gradient(u, x)
    return tf.concat((u, tf.multiply(-1.0,u_x)), 1)

def pinn_AL(f, x, training):
    with tf.GradientTape(persistent=True) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training) # shape = (k,) #! evaluate network                
        u_x = g2.gradient(u, x) # shape = (k,n) #! Calculate first derivative
    
    # https://github.com/tensorflow/tensorflow/issues/40885 -- batch_jacobian doesn't work with experimental compile
    # It does in 2.5, but there is a known issue for the implimentation: https://www.tensorflow.org/xla/known_issues?hl=nb
    # The while_loop needs to be bounded
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)
    laplacian = tf.reduce_sum(tf.linalg.diag_part(u_xx),1, keepdims=True)
    return tf.concat((tf.multiply(-1.0,u_x), laplacian),1)

def pinn_ALC(f, x, training):
    with tf.GradientTape(persistent=True) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training) # shape = (k,) #! evaluate network                
        u_x = g2.gradient(u, x) # shape = (k,n) #! Calculate first derivative
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)
    
    laplacian = tf.reduce_sum(tf.linalg.diag_part(u_xx),1, keepdims=True)

    curl_x = tf.math.subtract(u_xx[:,2,1], u_xx[:,1,2])
    curl_y = tf.math.subtract(u_xx[:,0,2], u_xx[:,2,0])
    curl_z = tf.math.subtract(u_xx[:,1,0], u_xx[:,0,1])

    curl = tf.stack([curl_x, curl_y, curl_z], axis=1)
    return tf.concat((tf.multiply(-1.0,u_x), laplacian, curl),1)

def pinn_APL(f, x, training):
    with tf.GradientTape(persistent=True) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training) # shape = (k,) #! evaluate network                
        u_x = g2.gradient(u, x) # shape = (k,n) #! Calculate first derivative
    
    # https://github.com/tensorflow/tensorflow/issues/40885 -- batch_jacobian doesn't work with experimental compile
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)
    laplacian = tf.reduce_sum(tf.linalg.diag_part(u_xx),1, keepdims=True)
    return tf.concat((u, tf.multiply(-1.0,u_x), laplacian),1)

def pinn_APLC(f, x, training):
    with tf.GradientTape(persistent=True) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training) # shape = (k,) #! evaluate network                
        u_x = g2.gradient(u, x) # shape = (k,n) #! Calculate first derivative
    u_xx = g1.batch_jacobian(u_x, x, experimental_use_pfor=True)
    
    laplacian = tf.reduce_sum(tf.linalg.diag_part(u_xx),1, keepdims=True)

    curl_x = tf.math.subtract(u_xx[:,2,1], u_xx[:,1,2])
    curl_y = tf.math.subtract(u_xx[:,0,2], u_xx[:,2,0])
    curl_z = tf.math.subtract(u_xx[:,1,0], u_xx[:,0,1])

    curl = tf.stack([curl_x, curl_y, curl_z], axis=1)

    return tf.concat((u,  tf.multiply(-1.0,u_x), laplacian, curl),1)