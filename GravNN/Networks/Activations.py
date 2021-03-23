import tensorflow as tf 

#@tf.function()
def leaky_relu(act_slope):
        return tf.keras.layers.LeakyReLU(alpha=act_slope)

@tf.function()
def bent_identity(x):
    return (tf.sqrt(tf.square(x)+1)-1)/2 + x

def radial_basis_function(x):
        return tf.exp(-tf.square(x))