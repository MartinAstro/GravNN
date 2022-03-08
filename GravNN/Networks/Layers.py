import tensorflow as tf
import numpy as np


class Cart2SphLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        """Layer responsible for taking cartesian inputs and converting
        them into spherical coordinates such that r in [0, inf),
        theta in [0,360], and phi in [0, 180].

        Note that this approach will leave discontinuities in the network output
        due to the discontinuities introduced by the atan2 function.

        Its successor is the Cart2PinesSph which bypasses the discontinuity entirely.
        """
        super(Cart2SphLayer, self).__init__()
        self.pi = tf.constant(np.pi, dtype=tf.float32).numpy()

    # @tf.function
    def call(self, inputs):
        pi = tf.constant(np.pi, dtype=tf.float32)
        inputs_transpose = tf.transpose(inputs)
        X = inputs_transpose[0]
        Y = inputs_transpose[1]
        Z = inputs_transpose[2]

        XX = tf.square(X)
        YY = tf.square(Y)
        r = tf.sqrt(XX + YY + tf.square(Z))  # r
        theta = tf.math.atan2(Y, X) + pi  # [rads] * 180.0 / pi + 180.0  # [0, 360]
        phi = tf.math.atan2(tf.sqrt(XX + YY), Z)  # [rads] * 180.0 / pi   # [0,180]

        spheres = tf.stack([r, theta, phi], axis=1)
        return spheres

    def get_config(self):

        config = super().get_config().copy()
        config.update(
            {
                "pi": self.pi,
            }
        )
        return config


class Sph2NetLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, scalers, mins):
        """Layer responsible for taking the spherical
        coordinates and normalizing them to a [-1,1]
        distribution.

        Args:
            input_dim (None): Invalid input (ignore) TODO: remove
            scalers (np.array): values used to scale each input quantity. These are computed before initialization
            mins (np.array): values used to bias each input before scaling. These are computed before initialization
        """
        super(Sph2NetLayer, self).__init__()
        self.scalers = tf.constant(scalers, dtype=tf.float32).numpy()
        self.mins = tf.constant(mins, dtype=tf.float32).numpy()

    # @tf.function
    def call(self, inputs):
        pi = tf.constant(np.pi, dtype=tf.float32)
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        theta = inputs_transpose[1]
        phi = inputs_transpose[2]

        # Normalize r -> [-1,1]
        r = tf.add(tf.multiply(r, self.scalers[0]), self.mins[0])
        # r = tf.divide(1.0,r)

        # Convert deg -> rad -> periodic [-1,1]
        theta = tf.sin(theta)
        phi = tf.cos(phi)
        spheres = tf.stack([r,theta, phi], axis=1)

        return spheres

    def get_config(self):
        config = super().get_config().copy()

        config.update(
            {
                "scalers": self.scalers,
                "mins": self.mins,
            }
        )
        config = super().get_config().copy()
        return config


class Cart2PinesSphLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        """Successor to the Cart2SphLayer. The layer takes a
        cartesian input and transforms it into a non-singular spherical
        representation (see Pines formulation). This bypasses the singularity introduced at the pole
        when taking a derivative of the potential.

        https://ntrs.nasa.gov/api/citations/19760011100/downloads/19760011100.pdf
        defines of alpha, beta, and gamma (i.e. three angle non-singular system)
        Args:
            input_dim (None): Invalid argument (ignore) TODO: remove
        """
        super(Cart2PinesSphLayer, self).__init__()

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
    def __init__(self, input_dim, scalers, mins, ref_radius):
        """Layer used to normalize the r value of the Cart2PinesSphLayer
        between the values of [-1,1]

        Args:
                input_dim (None): Invalid input (ignore) TODO: remove
                scalers (np.array): values used to scale each input quantity. These are computed before initialization
                mins (np.array): values used to bias each input before scaling. These are computed before initialization
        """
        super(PinesSph2NetLayer, self).__init__()
        self.scalers = tf.constant(scalers, dtype=tf.float32).numpy()
        self.mins = tf.constant(mins, dtype=tf.float32).numpy()
        self.ref_radius = tf.constant(ref_radius, dtype=tf.float32).numpy()

    # @tf.function
    def call(self, inputs):
        pi = tf.constant(np.pi, dtype=tf.float32)
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        beta = inputs_transpose[1]
        gamma = inputs_transpose[2]
        alpha = inputs_transpose[3]

        # Normalize r -> [-1,1]
        r = tf.add(tf.multiply(r, self.scalers[0]), self.mins[0])
        
        # Convert deg -> rad -> periodic [-1,1]
        # beta = tf.sin(beta)
        # gamma = tf.sin(gamma)
        # alpha = tf.sin(alpha)

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
    def __init__(self, input_dim, scalers, mins, ref_radius):
        """Layer used to normalize the r value of the Cart2PinesSphLayer
        between the values of [-1,1]

        Args:
                input_dim (None): Invalid input (ignore) TODO: remove
                scalers (np.array): values used to scale each input quantity. These are computed before initialization
                mins (np.array): values used to bias each input before scaling. These are computed before initialization
        """
        super(PinesSph2NetRefLayer, self).__init__()
        # self.ref_radius = tf.constant(ref_radius, dtype=tf.float32)#.numpy()
        self.ref_radius = ref_radius

    # @tf.function
    def call(self, inputs):
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        beta = inputs_transpose[1]
        gamma = inputs_transpose[2]
        alpha = inputs_transpose[3]

        # Normalize r -> [0,inf]
        r = tf.divide(self.ref_radius,r)
        spheres = tf.stack([r, beta, gamma, alpha], axis=1)
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
    def __init__(self, input_dim):
        """Layer used to normalize the r value of the Cart2PinesSphLayer
        between the values of [-1,1]

        Args:
                input_dim (None): Invalid input (ignore) TODO: remove
                scalers (np.array): values used to scale each input quantity. These are computed before initialization
                mins (np.array): values used to bias each input before scaling. These are computed before initialization
        """
        super(PinesSph2NetLayer_v2, self).__init__()

    # @tf.function
    def call(self, inputs):
        pi = tf.constant(np.pi, dtype=tf.float32)
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        s = inputs_transpose[1]
        t = inputs_transpose[2]
        u = inputs_transpose[3]

        r_inv = tf.divide(1,r)
        spheres = tf.stack([r_inv, s, t, u], axis=1)

        # r_inv = tf.divide(1,r)
        # r_tanh = tf.keras.activations.tanh(r)
        # spheres = tf.stack([r_inv, r_tanh, s, t, u], axis=1)

        # r_inv = tf.divide(1,r)
        # numerator = tf.math.log(r)
        # denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        # log_r = tf.keras.activations.tanh(numerator / denominator)
        # # r_tanh = tf.keras.activations.tanh(r)
        # spheres = tf.stack([r_inv, log_r, s, t, u], axis=1)
        return spheres

    def get_config(self):
        config = super().get_config().copy()
        return config