import tensorflow as tf
import numpy as np


class Cart2CylLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Cart2CylLayer, self).__init__()
        self.pi = tf.constant(np.pi, dtype=tf.float32).numpy()

    #@tf.function
    def call(self, inputs):
        pi = tf.constant(np.pi, dtype=tf.float32)
        inputs_transpose = tf.transpose(inputs)
        X = inputs_transpose[0]
        Y = inputs_transpose[1]
        Z = inputs_transpose[2]

        XX = tf.square(X)
        YY = tf.square(Y)
        r = tf.sqrt(XX + YY + tf.square(Z))  # r
        theta = tf.math.atan2(Y, X) + pi #[rads] * 180.0 / pi + 180.0  # [0, 360]

        r = tf.reshape(r, shape=[-1, 1])
        theta = tf.reshape(theta, shape=[-1, 1])
        Z = tf.reshape(Z, shape=[-1, 1])

        cylindrical = tf.concat([r, theta, Z], axis=-1)
        return cylindrical
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'pi': self.pi,
        })
        return config


class Cyl2NetLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, scalers, mins):
        super(Cyl2NetLayer, self).__init__()
        self.scalers = tf.constant(scalers, dtype=tf.float32).numpy()
        self.mins = tf.constant(mins, dtype=tf.float32).numpy()

    #@tf.function
    def call(self, inputs):
        pi = tf.constant(np.pi, dtype=tf.float32)
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        theta = inputs_transpose[1]
        Z = inputs_transpose[2]

        # Normalize r -> [-1,1]
        r = tf.add(tf.multiply(r,self.scalers[0]), self.mins[0])

        # Convert deg -> rad -> periodic [-1,1] 
        theta = tf.sin(theta)

        r = tf.reshape(r, shape=[-1, 1])
        theta = tf.reshape(theta, shape=[-1, 1])
        Z = tf.reshape(Z, shape=[-1, 1])

        cylindrical = tf.concat([r, theta, Z], axis=-1)
        return cylindrical
    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'scalers': self.scalers,
            'mins': self.mins,
        })
        config = super().get_config().copy()
        return config



class Cart2SphLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Cart2SphLayer, self).__init__()
        self.pi = tf.constant(np.pi, dtype=tf.float32).numpy()

    #@tf.function
    def call(self, inputs):
        pi = tf.constant(np.pi, dtype=tf.float32)
        inputs_transpose = tf.transpose(inputs)
        X = inputs_transpose[0]
        Y = inputs_transpose[1]
        Z = inputs_transpose[2]

        XX = tf.square(X)
        YY = tf.square(Y)
        r = tf.sqrt(XX + YY + tf.square(Z))  # r
        theta = tf.math.atan2(Y, X) + pi #[rads] * 180.0 / pi + 180.0  # [0, 360]
        phi = tf.math.atan2(tf.sqrt(XX + YY),Z) #[rads] * 180.0 / pi   # [0,180]

        r = tf.reshape(r, shape=[-1, 1])
        theta = tf.reshape(theta, shape=[-1, 1])
        phi = tf.reshape(phi, shape=[-1, 1])

        spheres = tf.concat([r, theta, phi], axis=-1)
        return spheres
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'pi': self.pi,
        })
        return config

class Sph2NetLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, scalers, mins):
        super(Sph2NetLayer, self).__init__()
        self.scalers = tf.constant(scalers, dtype=tf.float32).numpy()
        self.mins = tf.constant(mins, dtype=tf.float32).numpy()

    #@tf.function
    def call(self, inputs):
        pi = tf.constant(np.pi, dtype=tf.float32)
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        theta = inputs_transpose[1]
        phi = inputs_transpose[2]

        # Normalize r -> [-1,1]
        r = tf.add(tf.multiply(r,self.scalers[0]), self.mins[0])
        #r = tf.divide(1.0,r)

        # Convert deg -> rad -> periodic [-1,1] 
        theta = tf.sin(theta)
        phi = tf.cos(phi)

        r = tf.reshape(r, shape=[-1, 1])
        theta = tf.reshape(theta, shape=[-1, 1])
        phi = tf.reshape(phi, shape=[-1, 1])

        spheres = tf.concat([r, theta, phi], axis=-1)
        return spheres
    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'scalers': self.scalers,
            'mins': self.mins,
        })
        config = super().get_config().copy()
        return config



class Cart2PinesSphLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(Cart2PinesSphLayer, self).__init__()
        self.pi = tf.constant(np.pi, dtype=tf.float32).numpy()

    #@tf.function
    def call(self, inputs):
        pi = tf.constant(np.pi, dtype=tf.float32)
        inputs_transpose = tf.transpose(inputs)
        X = inputs_transpose[0]
        Y = inputs_transpose[1]
        Z = inputs_transpose[2]

        XX = tf.square(X)
        YY = tf.square(Y)
        r = tf.sqrt(XX + YY + tf.square(Z))  # r

        s = X/r # sin(beta)
        t = Y/r # sin(gamma)
        u = Z/r # sin(alpha)

        # beta = tf.math.asin(s)
        # gamma = tf.math.asin(t)
        # alpha = tf.math.asin(u)

        beta = s
        gamma = t
        alpha = u

        r = tf.reshape(r, shape=[-1, 1])
        beta = tf.reshape(beta, shape=[-1, 1])
        gamma = tf.reshape(gamma, shape=[-1, 1])
        alpha = tf.reshape(alpha, shape=[-1, 1])

        spheres = tf.concat([r, beta, gamma, alpha], axis=-1)
        return spheres
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'pi': self.pi,
        })
        return config

class PinesSph2NetLayer(tf.keras.layers.Layer):
    #https://ntrs.nasa.gov/api/citations/19760011100/downloads/19760011100.pdf
    # contains definitions of alpha, beta, and gamma
    # (three angle non-singular system)
    def __init__(self, input_dim, scalers, mins):
        super(PinesSph2NetLayer, self).__init__()
        self.scalers = tf.constant(scalers, dtype=tf.float32).numpy()
        self.mins = tf.constant(mins, dtype=tf.float32).numpy()

    #@tf.function
    def call(self, inputs):
        pi = tf.constant(np.pi, dtype=tf.float32)
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        beta = inputs_transpose[1]
        gamma = inputs_transpose[2]
        alpha = inputs_transpose[2]

        # Normalize r -> [-1,1]
        r = tf.add(tf.multiply(r,self.scalers[0]), self.mins[0])
        #r = tf.divide(1.0,r)

        # Convert deg -> rad -> periodic [-1,1] 
        # beta = tf.sin(beta)
        # gamma = tf.sin(gamma)
        # alpha = tf.sin(alpha)

        r = tf.reshape(r, shape=[-1, 1])
        beta = tf.reshape(beta, shape=[-1, 1])
        gamma = tf.reshape(gamma, shape=[-1, 1])
        alpha = tf.reshape(alpha, shape=[-1, 1])

        spheres = tf.concat([r, beta, gamma, alpha], axis=-1)

        return spheres
    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'scalers': self.scalers,
            'mins': self.mins,
        })
        config = super().get_config().copy()
        return config


class NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, scalers, mins):
        super(NormalizationLayer, self).__init__()
        self.pi = tf.constant(np.pi, dtype=tf.float32).numpy()
        self.scalers = tf.constant(scalers, dtype=tf.float32).numpy()
        self.mins = tf.constant(mins, dtype=tf.float32).numpy()
    #@tf.function
    def call(self, inputs):
        return tf.add(tf.multiply(inputs,self.scalers), self.mins)
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'pi': self.pi,
            'scalers': self.scalers,
            'mins': self.mins,
        })
        return config
# layer = Cart2SphLayer((1,3))
# print(layer([[10.0, 0., 0.]])) # Takes in cartesian vector



