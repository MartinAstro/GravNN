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

        r = tf.reshape(r, shape=[-1, 1])
        theta = tf.reshape(theta, shape=[-1, 1])
        phi = tf.reshape(phi, shape=[-1, 1])

        spheres = tf.concat([r, theta, phi], axis=-1)
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

        r = tf.reshape(r, shape=[-1, 1])
        theta = tf.reshape(theta, shape=[-1, 1])
        phi = tf.reshape(phi, shape=[-1, 1])

        spheres = tf.concat([r, theta, phi], axis=-1)
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

    # @tf.function
    def call(self, inputs):
        inputs_transpose = tf.transpose(inputs)
        X = inputs_transpose[0]
        Y = inputs_transpose[1]
        Z = inputs_transpose[2]

        XX = tf.square(X)
        YY = tf.square(Y)
        r = tf.sqrt(XX + YY + tf.square(Z))  # r

        s = X / r  # sin(beta)
        t = Y / r  # sin(gamma)
        u = Z / r  # sin(alpha)

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
        #r = tf.add(tf.multiply(r, self.scalers[0]), self.mins[0])

        # (R/r) - 1 where R is the Brill Radius of Body
        r = tf.subtract(tf.divide(1,r), 1)
        # r = tf.divide(1.0,r)

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

        config.update(
            {
                "scalers": self.scalers,
                "mins": self.mins,
                "ref_radius":self.ref_radius
            }
        )
        config = super().get_config().copy()
        return config
