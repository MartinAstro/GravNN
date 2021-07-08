import tensorflow as tf

def snake(x):
    return tf.add(x, tf.square(tf.sin(x)))