from asyncio import constants
import tensorflow as tf
import numpy as np


class PreprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, min, scale, dtype):
        super(PreprocessingLayer, self).__init__(dtype=dtype)
        self.min = tf.constant(min, dtype=dtype)
        self.scale = tf.constant(scale, dtype=dtype)

    def call(self, inputs):
        normalized_inputs = inputs * self.scale + self.min
        return normalized_inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "min": self.min,
                "scale": self.scale,
            }
        )
        return config

class PostprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, min, scale, dtype):
        super(PostprocessingLayer, self).__init__(dtype=dtype)
        self.min = tf.constant(min, dtype=dtype)
        self.scale = tf.constant(scale, dtype=dtype)

    def call(self, normalized_inputs):
        inputs = (normalized_inputs - self.min)/self.scale
        return inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "min": self.min,
                "scale": self.scale,
            }
        )
        return config

class Cart2PinesSphLayer(tf.keras.layers.Layer):
    def __init__(self, dtype):
        """Successor to the Cart2SphLayer. The layer takes a
        cartesian input and transforms it into a non-singular spherical
        representation (see Pines formulation). This bypasses the singularity introduced at the pole
        when taking a derivative of the potential.

        https://ntrs.nasa.gov/api/citations/19760011100/downloads/19760011100.pdf
        defines of alpha, beta, and gamma (i.e. three angle non-singular system)
        """
        super(Cart2PinesSphLayer, self).__init__(dtype=dtype)

    def call(self, inputs):
        inputs_transpose = tf.transpose(inputs)
        X = inputs_transpose[0]
        Y = inputs_transpose[1]
        Z = inputs_transpose[2]

        XX = tf.square(X)
        YY = tf.square(Y)
        ZZ = tf.square(Z)
        r = tf.sqrt(XX + YY + ZZ)  # r

        s = X / r  # sin(beta)
        t = Y / r  # sin(gamma)
        u = Z / r  # sin(alpha)

        spheres = tf.stack([r, s, t, u], axis=1)
        return spheres

    def get_config(self):
        config = super().get_config().copy()
        return config

class PinesSph2NetLayer(tf.keras.layers.Layer):
    def __init__(self, dtype, scalers, mins, ref_radius):
        """Layer used to normalize the r value of the Cart2PinesSphLayer
        between the values of [-1,1]

        Args:
                scalers (np.array): values used to scale each input quantity. These are computed before initialization
                mins (np.array): values used to bias each input before scaling. These are computed before initialization
        """
        super(PinesSph2NetLayer, self).__init__(dtype=dtype)
        self.scalers = tf.constant(scalers, dtype=dtype).numpy()
        self.mins = tf.constant(mins, dtype=dtype).numpy()
        self.ref_radius = tf.constant(ref_radius, dtype=dtype).numpy()

    def call(self, inputs):
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        beta = inputs_transpose[1]
        gamma = inputs_transpose[2]
        alpha = inputs_transpose[3]

        # Normalize r -> [-1,1]
        r = tf.add(tf.multiply(r, self.scalers[0]), self.mins[0])
        spheres = tf.stack([r, beta, gamma, alpha], axis=1)
        return spheres

    def get_config(self):
        config = super().get_config().copy()

        config.update(
            {
                "scalers": self.scalers,
                "mins": self.mins,
                "ref_radius":self.ref_radius
            }
        )
        config = super().get_config().copy()
        return config

class PinesSph2NetRefLayer(tf.keras.layers.Layer):
    def __init__(self, dtype, scalers, mins, ref_radius):
        """Layer used to normalize the r value of the Cart2PinesSphLayer
        between the values of [-1,1]

        Args:
                scalers (np.array): values used to scale each input quantity. These are computed before initialization
                mins (np.array): values used to bias each input before scaling. These are computed before initialization
        """
        super(PinesSph2NetRefLayer, self).__init__(dtype=dtype)
        self.ref_radius = ref_radius

    def call(self, inputs):
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        beta = inputs_transpose[1]
        gamma = inputs_transpose[2]
        alpha = inputs_transpose[3]

        # Normalize r -> [0,inf]
        r_inv_prime = tf.divide(self.ref_radius,r)
        spheres = tf.stack([r_inv_prime, beta, gamma, alpha], axis=1)
        return spheres

    def get_config(self):
        config = super().get_config().copy()

        config.update(
            {
                "ref_radius":self.ref_radius
            }
        )
        config = super().get_config().copy()
        return config

class PinesSph2NetLayer_v2(tf.keras.layers.Layer):
    def __init__(self, dtype, ref_radius_min=None, ref_radius_max=None):
        super(PinesSph2NetLayer_v2, self).__init__(dtype=dtype)
        self.ref_radius_min = tf.cond(ref_radius_min != None, 
                        lambda : tf.constant(ref_radius_min, dtype=dtype), 
                        lambda : tf.constant(0.0, dtype=dtype)).numpy()
        self.ref_radius_max = tf.cond(ref_radius_max != None, 
                        lambda : tf.constant(ref_radius_max, dtype=dtype), 
                        lambda : tf.constant(1.0, dtype=dtype)).numpy()
        
        # bounds of the feature -- shouldn't necessarily be a large difference.
        # think about how much the output should change with respect to the inputs
        self.s_min = tf.constant(1.0, dtype=dtype).numpy()
        self.s_max = tf.constant(1.0 + (self.ref_radius_max - self.ref_radius_min), dtype=dtype).numpy()           

    def call(self, inputs):
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        s = inputs_transpose[1]
        t = inputs_transpose[2]
        u = inputs_transpose[3]

        scale = tf.divide(self.s_max - self.s_min, self.ref_radius_max - self.ref_radius_min)
        min_arg = self.s_min - tf.multiply(self.ref_radius_min, scale)
        r_star = tf.multiply(r, scale) + min_arg

        one = tf.constant(1.0, dtype=r.dtype)
        r_inv = tf.divide(one, r_star)
        # r_inv = r_star
        spheres = tf.stack([r_inv, s, t, u], axis=1)
        return spheres

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "dtype" : self.dtype,
                "s_min" : self.s_min,
                "s_max" : self.s_max,
                "ref_radius_min" : self.ref_radius_min,
                "ref_radius_max" : self.ref_radius_max
            }
        )
        return config