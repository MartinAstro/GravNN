
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
    u_xx = g1.batch_jacobian(u_x, x)
    laplacian = tf.reduce_sum(tf.linalg.diag_part(u_xx),1, keepdims=True)
    return tf.concat((tf.multiply(-1.0,u_x), laplacian),1)

def pinn_ALC(f, x, training):
    with tf.GradientTape(persistent=True) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training) # shape = (k,) #! evaluate network                
        u_x = g2.gradient(u, x) # shape = (k,n) #! Calculate first derivative
    u_xx = g1.batch_jacobian(u_x, x)
    
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
    u_xx = g1.batch_jacobian(u_x, x)
    laplacian = tf.reduce_sum(tf.linalg.diag_part(u_xx),1, keepdims=True)
    return tf.concat((u, tf.multiply(-1.0,u_x), laplacian),1)

def pinn_APLC(f, x, training):
    with tf.GradientTape(persistent=True) as g1:
        g1.watch(x)
        with tf.GradientTape() as g2:
            g2.watch(x)
            u = f(x, training) # shape = (k,) #! evaluate network                
        u_x = g2.gradient(u, x) # shape = (k,n) #! Calculate first derivative
    u_xx = g1.batch_jacobian(u_x, x)
    
    laplacian = tf.reduce_sum(tf.linalg.diag_part(u_xx),1, keepdims=True)

    curl_x = tf.math.subtract(u_xx[:,2,1], u_xx[:,1,2])
    curl_y = tf.math.subtract(u_xx[:,0,2], u_xx[:,2,0])
    curl_z = tf.math.subtract(u_xx[:,1,0], u_xx[:,0,1])

    curl = tf.stack([curl_x, curl_y, curl_z], axis=1)

    return tf.concat((u,  tf.multiply(-1.0,u_x), laplacian, curl),1)




#def compute_spherical_gradient():
    # if self.config['basis'][0] == 'spherical':
    #     # This cannot work as currently designed. The gradient at theta [0, 180] is divergent. 
    #     with tf.GradientTape() as tape:
    #         tape.watch(x)
    #         U_pred = self.network(x, training)
    #     gradients = tape.gradient(U_pred, x)
    #     # https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates#Del_formula
    #     a0 = -gradients[:,0]
    #     # In wiki article, theta is 0-180 deg (which has been phi in our definition)
    #     theta = tf.add(tf.multiply(x[:,2],np.pi), np.pi)
    #     a1 = -(1.0/x[:,0])*(1.0/tf.sin(theta))*gradients[:,1]
    #     a2 = -(1.0/x[:,0])*gradients[:,2]

    #     #print(a2.shape)
    #     a_pred = tf.concat([[a0], [a1], [a2]], 0)
    #     a_pred = tf.reshape(a_pred, [-1, 3])
    # else:   
    # 

# # Periodic boundary conditions 
# if self.config['basis'][0] == 'spherical':
    
#     x_periodic = tf.add(x, [0, 2, 2])
#     U_pred_periodic, a_pred_periodic = self(x_periodic, training=True)
#     #a_pred_periodic = tf.where(tf.math.is_inf(a_pred_periodic), y, a_pred_periodic)
#     loss += self.compiled_loss(y, a_pred_periodic)

#     x_periodic = tf.add(x, [0, -2, -2])
#     U_pred_periodic, a_pred_periodic = self(x_periodic, training=True)
#     #a_pred_periodic = tf.where(tf.math.is_inf(a_pred_periodic), y, a_pred_periodic)
#     loss += self.compiled_loss(y, a_pred_periodic)

#     # 0 potential at infinity. 
#     x_infinite = tf.multiply(x, [1E308, 1, 1])
#     U_pred_infinite, a_pred_infinite = self(x_infinite, training=True)
#     a_pred_infinite = tf.where(tf.math.is_inf(a_pred_infinite), y, a_pred_infinite)
#     a_pred_infinite = tf.where(tf.math.is_nan(a_pred_infinite), y, a_pred_infinite)
#     loss += self.compiled_loss(tf.zeros_like(a_pred_infinite), a_pred_infinite)
