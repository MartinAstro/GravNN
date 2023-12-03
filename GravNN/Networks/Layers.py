import numpy as np
import tensorflow as tf

from GravNN.Networks.Losses import norm


def get_preprocess_layer_fcn(layer_key):
    return {
        "pines": Cart2PinesSphLayer,
        "r_scale": ScaleRLayer,
        "r_normalize": NormalizeRLayer,
        "r_inv": InvRLayer,
        "fourier": FourierFeatureLayer,
    }[layer_key.lower()]


# helper functions
def blend(x, y, z, k, x_ref):
    dr = tf.subtract(x, x_ref)
    h = 0.5 + 0.5 * tf.tanh(k * dr)
    output = (1.0 - h) * y + h * z
    return output


def H(x, r, k):
    # assuming k > 0 this goes from 0 -> 1 as x->inf
    dr = tf.subtract(x, r)
    h = 0.5 + 0.5 * tf.tanh(k * dr)
    return h


def G(x, r, k):
    # assuming k > 0 this goes from 1 -> 0 as x->inf
    return 1.0 - H(x, r, k)


def Hss(x, r_min, r_max):
    h = smooth_step(x, r_min, r_max)
    return h


def Gss(x, r_min, r_max):
    g = 1.0 - Hss(x, r_min, r_max)
    return g


def fuse(r, a, b, r0, k):
    h = H(r, r0, k)
    g = G(r, r0, k)
    return g * a + h * b


def r_safety_set(r, clip=1.0):
    r_inv = tf.math.divide_no_nan(1.0, r)
    r_inv_cap = tf.clip_by_value(r_inv, 0.0, clip)
    r_cap = tf.clip_by_value(r, 0.0, clip)
    return r_cap, r_inv_cap


def linear_map(r, a, b):
    return (r - a) / (b - a)


def smooth_step(r, r_ref_min, r_ref_max):
    x = linear_map(r, r_ref_min, r_ref_max)
    x = tf.clip_by_value(x, 0.0, 1.0)
    # phi = 3 * tf.pow(x, 2) - 2 * tf.pow(x, 3)
    phi_fifth_order = 6 * tf.pow(x, 5) - 15 * tf.pow(x, 4) + 10 * tf.pow(x, 3)

    # phi is only valid in the domain of x = [0, 1]
    # phi = tf.where(x > 1.0, 1.0, phi)
    # phi = tf.where(x < 0.0, 0.0, phi)
    # phi = tf.where(phi < 0.0, 0.0, phi)
    # phi = tf.where(phi > 1.0, 1.0, phi)
    return phi_fifth_order


def blend_smooth(r, f, g, r_ref_min=0.0, r_ref_max=1.0):
    # when phi = 0.0, exclusively use f
    # when phi = 1.0, exclusively use g
    phi = smooth_step(r, r_ref_min, r_ref_max)
    return (1.0 - phi) * f + phi * g


def compute_shape_parameters(**kwargs):
    R = kwargs["planet"][0].radius
    R_min = kwargs.get("ref_radius_min", [0.0])[0]
    x_transformer = kwargs["x_transformer"][0]

    R_vec = np.array([[R, R_min, 0]])
    R_vec_ND = x_transformer.transform(R_vec)

    a = R_vec_ND[0, 0]
    b = R_vec_ND[0, 1]

    e = np.sqrt(1 - b**2 / a**2)
    return a, b, e


def internal_scale(r_cap, a):
    r_sq = r_cap**2
    R_cubed = a**3
    scale = (r_sq / R_cubed) - (2.0 / a)
    return scale


# preprocessing
class PreprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, min, scale, dtype):
        super(PreprocessingLayer, self).__init__(dtype=dtype)
        self.min = tf.constant(min, dtype=dtype).numpy()
        self.scale = tf.constant(scale, dtype=dtype).numpy()

    def call(self, inputs):
        normalized_inputs = tf.math.add(tf.math.multiply(inputs, self.scale), self.min)
        return normalized_inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "min": self.min,
                "scale": self.scale,
            },
        )
        return config


class PostprocessingLayer(tf.keras.layers.Layer):
    def __init__(self, min, scale, dtype):
        super(PostprocessingLayer, self).__init__(dtype=dtype)
        self.min = tf.constant(min, dtype=dtype).numpy()
        self.scale = tf.constant(scale, dtype=dtype).numpy()

    def call(self, normalized_inputs):
        inputs = tf.math.divide(tf.subtract(normalized_inputs, self.min), self.scale)
        return inputs

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "min": self.min,
                "scale": self.scale,
            },
        )
        return config


class Cart2PinesSphLayer(tf.keras.layers.Layer):
    def __init__(self, dtype, **kwargs):
        """Successor to the Cart2SphLayer. The layer takes a
        cartesian input and transforms it into a non-singular spherical
        representation (see Pines formulation). This bypasses the singularity introduced
        at the pole when taking a derivative of the potential.

        https://ntrs.nasa.gov/api/citations/19760011100/downloads/19760011100.pdf
        defines of alpha, beta, and gamma (i.e. three angle non-singular system)
        """
        super(Cart2PinesSphLayer, self).__init__(dtype=dtype)

    def call(self, inputs):
        # s = X / r  # sin(beta)
        # t = Y / r  # sin(gamma)
        # u = Z / r  # sin(alpha)

        r = norm(inputs)
        stu = tf.math.divide_no_nan(inputs, r)
        spheres = tf.concat([r, stu], axis=1)
        return spheres

    def get_config(self):
        config = super().get_config().copy()
        return config


class InvRLayer(tf.keras.layers.Layer):
    def __init__(self, dtype, **kwargs):
        super(InvRLayer, self).__init__(dtype=dtype)

    def call(self, inputs):
        r = inputs[:, 0:1]
        r_cap, r_inv_cap = r_safety_set(r)
        spheres = tf.concat([r_cap, r_inv_cap, inputs[:, 1:4]], axis=1)
        return spheres

    def get_config(self):
        config = super().get_config().copy()
        return config


# postprocessing
class AnalyticModelLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        dtype = kwargs.get("dtype")[0]
        super(AnalyticModelLayer, self).__init__(dtype=dtype)

        # defaults to zero
        self.mu = kwargs.get("mu_non_dim", [0.0])[0]
        self.C20 = kwargs.get("cBar", [np.zeros((3, 3))])[0][2, 0]

        transition_potential = kwargs.get("use_transition_potential", [True])[0]
        self.use_transition_potential = transition_potential

        self.c1 = np.sqrt(15.0 / 4.0) * np.sqrt(3.0)
        self.c2 = np.sqrt(5.0 / 4.0)

        # ensure proper dtype
        self.mu = tf.constant(self.mu, dtype=dtype).numpy()
        self.C20 = tf.constant(self.C20, dtype=dtype).numpy()
        self.c1 = tf.constant(self.c1, dtype=dtype).numpy()
        self.c2 = tf.constant(self.c2, dtype=dtype).numpy()
        self.use_transition_potential = tf.constant(
            self.use_transition_potential,
            dtype=tf.bool,
        ).numpy()

        a, b, e = compute_shape_parameters(**kwargs)
        self.a = tf.constant(a, dtype=dtype).numpy()
        self.b = tf.constant(b, dtype=dtype).numpy()

        self.trainable_tanh = kwargs.get("trainable_tanh", [True])[0]

    def build(self, input_shapes):
        self.k_external = self.add_weight(
            "k_external",
            shape=[1],
            trainable=False,
            initializer=tf.keras.initializers.Constant(value=0.5),
        )
        self.r_external = self.add_weight(
            "r_external",
            shape=[1],
            trainable=False,
            initializer=tf.keras.initializers.Constant(value=0.0),
        )
        super(AnalyticModelLayer, self).build(input_shapes)

    def call(self, inputs):
        r = inputs[:, 0:1]
        u = inputs[:, 3:4]

        r_cap, r_inv_cap = r_safety_set(r)

        # External
        # Compute point mass approximation assuming
        u_pm_external = self.mu * r_inv_cap
        u_C20 = (
            (self.a * r_inv_cap) ** 2
            * u_pm_external
            * (u**2 * self.c1 - self.c2)
            * self.C20
        )
        u_external_full = tf.negative(u_pm_external + u_C20)

        # Internal
        u_external_pm_boundary = self.mu / self.a
        u_external_C20_boundary = (
            u_external_pm_boundary * (u**2 * self.c1 - self.c2) * self.C20
        )
        u_boundary = tf.negative(u_external_pm_boundary + u_external_C20_boundary)
        u_internal = self.mu * (r_cap**2 / self.a**3) + 2 * u_boundary

        u_analytic = tf.where(r < self.a, u_internal, u_external_full)

        # decrease the weight of the model in the region between
        # the interior of the asteroid and out to r < 1 + e, where
        # e is the eccentricity of the asteroid geometry, because
        # in this regime, a point mass / SH assumption adds unnecessary
        # error.
        h_external = H(r, self.r_external, self.k_external)
        u_analytic = u_analytic * h_external

        return u_analytic

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "mu": self.mu,
                "a": self.a,
                "C20": self.C20,
                "c1": self.c1,
                "c2": self.c2,
                "use_transition_potential": self.use_transition_potential,
            },
        )
        return config


class NetworkBoundingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        dtype = kwargs.get("dtype")[0]
        super(NetworkBoundingLayer, self).__init__(dtype=dtype)

        # defaults to zero
        self.mu = kwargs.get("mu_non_dim", [0.0])[0]

        # ensure proper dtype
        self.mu = tf.constant(self.mu, dtype=dtype).numpy()

        a, b, e = compute_shape_parameters(**kwargs)
        self.b = tf.constant(b, dtype=dtype).numpy()

    def call(self, u_nn):
        # Never let the neural network produce a value of the potential that exceeds
        # mu / semi-minor axis. This is a hard constraint that must be satisfied.
        u_nn_bound = tf.clip_by_value(u_nn, -self.mu / self.b, self.mu / self.b)
        return u_nn_bound

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "mu": self.mu,
                "a": self.b,
            },
        )
        return config


class ScaleNNPotential(tf.keras.layers.Layer):
    def __init__(self, power, **kwargs):
        dtype = kwargs["dtype"][0]
        super(ScaleNNPotential, self).__init__(dtype=dtype)
        self.power = tf.constant(power, dtype=dtype).numpy()
        use_transition_potential = kwargs.get("use_transition_potential", [True])[0]
        self.use_transition_potential = tf.constant(
            use_transition_potential,
            dtype=tf.bool,
        ).numpy()

        self.scale_potential = kwargs.get("scale_nn_potential", [True])[0]
        a, b, e = compute_shape_parameters(**kwargs)
        self.a = tf.constant(a, dtype=dtype).numpy()
        self.b = tf.constant(b, dtype=dtype).numpy()
        self.e = tf.constant(e, dtype=dtype).numpy()

    def call(self, features, u_nn):
        r = features[:, 0:1]
        r_cap, r_inv_cap = r_safety_set(r)

        if not self.scale_potential:
            return u_nn

        # scale the internal potential
        # scale = internal_scale(r_cap, self.a)
        # u_scaled_internal = tf.negative(u_nn * scale)

        # scale the external potential down to correct order of mag
        # U = U_NN * 1 / r^power
        scale_external = tf.pow(r_inv_cap, self.power)

        # Don't scale within the critical radius (1+e)
        tf.ones_like(scale_external)

        # Must use a smooth_step function instead of tanh to
        # force solution scaling to 0 or 1.
        # R_trans = 1.0 + self.e
        # scale = blend_smooth(r, scale_internal, scale_external, R_trans, 2*R_trans)
        # u_final = u_nn * scale
        u_final = u_nn * scale_external

        return u_final

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "power": self.power,
                "use_transition_potential": self.use_transition_potential,
                "a": self.a,
                "b": self.b,
                "e": self.e,
            },
        )
        return config


class FuseModels(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        dtype = kwargs["dtype"][0]
        fuse_models = kwargs["fuse_models"][0]
        super(FuseModels, self).__init__(dtype=dtype)
        self.fuse = tf.constant(int(fuse_models), dtype=dtype).numpy()

    def call(self, u_nn, u_analytic):
        fuse_vector = tf.constant(self.fuse, dtype=u_nn.dtype)
        u = u_nn + fuse_vector * u_analytic
        return u

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "fuse": self.fuse,
            },
        )
        return config


class EnforceBoundaryConditions(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        dtype = kwargs["dtype"][0]
        super(EnforceBoundaryConditions, self).__init__(dtype=dtype)
        self.enforce_bc = kwargs["enforce_bc"][0]
        r_max = kwargs.get("ref_radius_analytic", [np.nan])[0]
        self.trainable_tanh = kwargs.get("trainable_tanh")[0]
        # self.r_max = kwargs.get("tanh_r", [r_max])[0]
        # self.k_init = kwargs.get("tanh_k", [1.0])[0]
        # self.r_max -= 1.0

        # if kwargs['fuse_models']:
        #     self.k_init = 1.0/ self.k_init

        self.r_max = kwargs.get("tanh_r", [r_max])[0]
        self.k_init = kwargs.get("tanh_k", [1.0])[0]

    def build(self, input_shapes):
        self.radius = self.add_weight(
            "radius",
            shape=[1],
            trainable=self.trainable_tanh,
            initializer=tf.keras.initializers.Constant(value=self.r_max),
        )
        self.k = self.add_weight(
            "k",
            shape=[1],
            trainable=self.trainable_tanh,
            initializer=tf.keras.initializers.Constant(value=self.k_init),
        )
        super(EnforceBoundaryConditions, self).build(input_shapes)

    def call(self, features, u_nn, u_analytic):
        if not self.enforce_bc:
            return u_nn
        r = features[:, 0:1]
        h = H(r, self.radius, self.k)
        g = G(r, self.radius, self.k)
        u_model = g * u_nn + h * u_analytic
        return u_model

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "enforce_bc": self.enforce_bc,
                "trainable_tanh": self.trainable_tanh,
                "k": self.k_init,
                "r_max": self.r_max,
            },
        )
        return config


# Experimental
class FourierFeatureLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        fourier_features,
        fourier_sigma,
        freq_decay,
        trainable,
        shared_freq,
        shared_offset,
        sine_and_cosine,
        base_2_init,
        dtype,
        **kwargs,
    ):
        super(FourierFeatureLayer, self).__init__(dtype=dtype)

        self.fourier_features = fourier_features
        self.fourier_sigma = fourier_sigma
        self.freq_decay = freq_decay
        self.trainable = trainable
        self.shared_freq = shared_freq
        self.shared_offset = shared_offset
        self.sine_and_cosine = sine_and_cosine
        self.base_2_init = base_2_init

    def build(self, input_shapes):
        features = self.fourier_features
        if self.sine_and_cosine:
            features = self.fourier_features // 2

        freqs_list = []
        freqs = None
        for i in range(3):
            # if each angle shares same freq reappend same variables
            if self.shared_freq and i > 0:
                freqs_list.append(freqs)
                continue

            # default to random initializer
            freq_initializer = tf.keras.initializers.RandomNormal(
                mean=tf.constant(0.0, dtype=self.dtype),
                stddev=tf.constant(self.fourier_sigma, dtype=self.dtype),
                seed=1234,
            )
            if self.base_2_init:
                init_base_2 = tf.constant(
                    [2**i for i in range(0, self.fourier_features)],
                    dtype=self.dtype,
                )
                freq_initializer = tf.keras.initializers.Constant(init_base_2)

            # otherwise, generate new variables and append
            freqs = self.add_weight(
                f"freq_{i}",
                shape=[features, 1],
                trainable=self.trainable,
                initializer=freq_initializer,
            )
            freqs_list.append(freqs)

        offsets_list = []
        phase_offset = None
        for i in range(3):
            # if each angle shares same freq reappend same variables
            if self.shared_offset and i > 0:
                offsets_list.append(phase_offset)
                continue

            # otherwise, generate new variables and append
            phase_offset = self.add_weight(
                f"phase_{i}",
                shape=[features],
                trainable=self.trainable,
                initializer=tf.keras.initializers.Zeros(),
            )
            offsets_list.append(phase_offset)

        self.s_freq = tf.transpose(freqs_list[0])
        self.t_freq = tf.transpose(freqs_list[1])
        self.u_freq = tf.transpose(freqs_list[2])

        self.s_offset = offsets_list[0]
        self.t_offset = offsets_list[1]
        self.u_offset = offsets_list[2]

        super(FourierFeatureLayer, self).build(input_shapes)

    def call(self, inputs):
        one = tf.constant(1.0, dtype=self.dtype)
        two = tf.constant(2.0, dtype=self.dtype)
        two_pi = tf.constant(2 * np.pi, dtype=self.dtype)

        r = inputs[:, 0:1]  # [N x 1]
        s = inputs[:, 1:2]
        t = inputs[:, 2:3]
        u = inputs[:, 3:4]

        def project(x, x_freq, x_offset):
            # force geometry to be between 0 - 1
            x_mod = (x + one) / two
            # change to rad
            x_mod *= two_pi
            # multiply by freq + offset phase
            x_FF = x_mod @ x_freq + x_offset
            # optionally scale by freq + altitude
            if self.freq_decay:
                x_r_scale = tf.math.pow(r, x_freq)
            else:
                x_r_scale = tf.ones_like(x_FF)
            # fourier projection
            x_sin = x_r_scale * tf.sin(x_FF)
            x_cos = x_r_scale * tf.cos(x_FF)
            return x_sin, x_cos

        s_sin, s_cos = project(s, self.s_freq, self.s_offset)
        t_sin, t_cos = project(t, self.t_freq, self.t_offset)
        u_sin, u_cos = project(u, self.u_freq, self.u_offset)

        stu_sin = tf.concat([s_sin, t_sin, u_sin], axis=1)
        stu_cos = tf.concat([s_cos, t_cos, u_cos], axis=1)

        # stack radius and fourier basis together
        if self.sine_and_cosine:
            features = tf.concat([r, s, t, u, stu_sin, stu_cos], 1)  # [N x 2M+1]
        else:
            features = tf.concat([r, s, t, u, stu_sin], 1)  # [N x 2M+1]

        return features

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "fourier_features": self.fourier_features,
                "fourier_sigma": self.fourier_sigma,
                "freq_decay": self.freq_decay,
                "trainable": self.trainable,
                "shared_freq": self.shared_freq,
                "shared_offset": self.shared_offset,
                "sine_and_cosine": self.sine_and_cosine,
                "base_2_init": self.base_2_init,
            },
        )
        return config


# Legacy
class PinesAlgorithmLayer(tf.keras.layers.Layer):
    def __init__(self, dtype, mu, a, cBar, sBar):
        super(PinesAlgorithmLayer, self).__init__(dtype=dtype)
        self.mu = tf.constant(mu, dtype=dtype).numpy()
        self.a = tf.constant(a, dtype=dtype).numpy()
        self.cBar = tf.constant(cBar, dtype=dtype).numpy()
        self.sBar = tf.constant(sBar, dtype=dtype).numpy()
        self.N = tf.constant(len(cBar) - 3, dtype=tf.int32).numpy()
        self.n1, self.n2 = self.compute_normalization_constants(self.N)
        # a = self.compute_aBar(tf.constant(10,dtype=dtype))
        # rE, iM = self.compute_rE_iM(
        #           tf.constant(10,dtype=dtype),
        #           tf.constant(10,dtype=dtype))

    def getK(self, x):
        return (
            tf.constant(1.0, dtype=self.dtype)
            if (x == 0)
            else tf.constant(2.0, dtype=self.dtype)
        )

    def compute_normalization_constants(self, N):
        n1 = tf.zeros((N + 2, N + 2), dtype=self.dtype)
        n2 = tf.zeros((N + 2, N + 2), dtype=self.dtype)

        for l_idx in range(0, N + 2):
            for m in range(0, l_idx + 1):
                if l_idx >= m + 2:
                    l = tf.constant([l_idx], self.dtype)  # noqa: E741
                    n1_lm = tf.sqrt(
                        ((2.0 * l + 1.0) * (2.0 * l - 1.0)) / ((l - m) * (l + m)),
                    )
                    n2_lm = tf.sqrt(
                        ((l + m - 1.0) * (2.0 * l + 1.0) * (l - m - 1.0))
                        / ((l + m) * (l - m) * (2.0 * l - 3.0)),
                    )
                    n1 = tf.tensor_scatter_nd_update(
                        n1,
                        [[l_idx, m]],
                        n1_lm,
                        name="n1_update",
                    )
                    n2 = tf.tensor_scatter_nd_update(
                        n2,
                        [[l_idx, m]],
                        n2_lm,
                        name="n2_update",
                    )

        return n1.numpy(), n2.numpy()

    def compute_rE_iM(self, s, t):
        rE = tf.scatter_nd(
            tf.constant([[0]]),
            tf.constant([1.0], dtype=self.dtype),
            shape=tf.constant([self.N + 2]),
            name="rE",
        )
        iM = tf.scatter_nd(
            tf.constant([[0]]),
            tf.constant([0.0], dtype=self.dtype),
            shape=tf.constant([self.N + 2]),
            name="iM",
        )

        for i in range(1, self.N + 2):
            rE_m1 = tf.gather(rE, [i - 1])
            iM_m1 = tf.gather(iM, [i - 1])
            rE = tf.tensor_scatter_nd_update(
                rE,
                [[i]],
                s * rE_m1 - t * iM_m1,
                name="rE_update",
            )  # introduces error
            iM = tf.tensor_scatter_nd_update(
                iM,
                [[i]],
                s * iM_m1 + t * rE_m1,
                name="iM_update",
            )

        return rE, iM

    def compute_aBar(self, u):
        N = self.N
        aBar = tf.scatter_nd(
            [[0, 0]],
            tf.constant([1.0], dtype=u.dtype),
            shape=((N + 2, N + 2)),
            name="aBar",
        )

        for l in tf.range(1, N + 2):  # noqa: E741
            a_lm1_lm1 = tf.gather_nd(aBar, [[l - 1, l - 1]], name="aBar_gather")
            l_float = tf.cast(l, dtype=self.dtype)
            a_l_l = (
                tf.sqrt(
                    (2.0 * l_float + 1.0)
                    * self.getK(l)
                    / (2.0 * l_float * self.getK(l - 1)),
                )
                * a_lm1_lm1
            )
            a_l_lm1 = (
                tf.sqrt(2.0 * l_float * self.getK(l - 1) / self.getK(l)) * a_l_l * u
            )
            aBar = tf.tensor_scatter_nd_update(
                aBar,
                [[l, l]],
                a_l_l,
                name="aBar_update_1",
            )
            aBar = tf.tensor_scatter_nd_update(
                aBar,
                [[l, l - 1]],
                a_l_lm1,
                name="aBar_update_2",
            )

        for m in range(0, N + 2):
            for l in range(m + 2, N + 2):  # noqa: E741
                a_lm1_m = tf.gather_nd(aBar, [[l - 1, m]], name="a_lm1_m")
                a_lm2_m = tf.gather_nd(aBar, [[l - 2, m]], name="a_lm2_m")
                n1_lm = tf.gather_nd(self.n1, [[l, m]], name="n1_lm")
                n2_lm = tf.gather_nd(self.n2, [[l, m]], name="n2_lm")
                a_lm = u * n1_lm * a_lm1_m - n2_lm * a_lm2_m
                aBar = tf.tensor_scatter_nd_update(
                    aBar,
                    [[l, m]],
                    a_lm,
                    name="aBar_final",
                )

        return aBar

    def compute_rhol(self, a, r):
        rho = a / r
        rhol = tf.zeros((self.N + 1), dtype=self.dtype)
        rhol = tf.scatter_nd([[0]], [self.mu / r], shape=((self.N + 1,)), name="rho")
        rhol = tf.tensor_scatter_nd_update(
            rhol,
            [[0]],
            [self.mu / r],
            name="rho_update_0",
        )  # good
        rhol = tf.tensor_scatter_nd_update(
            rhol,
            [[1]],
            [self.mu / r * rho],
            name="rho_update_1",
        )  # good
        for l in range(1, self.N):  # noqa: E741
            rho_i = tf.gather(rhol, [l], name="rho_gather")
            rhol = tf.tensor_scatter_nd_update(
                rhol,
                [[l + 1]],
                rho * rho_i,
                name="rho_update_2",
            )  # introduce error
        return rhol

    def compute_potential(self, r, s, t, u, a):
        rE, iM = self.compute_rE_iM(s, t)
        rhol = self.compute_rhol(a, r)
        aBar = self.compute_aBar(u)

        potential = 0.0
        for l in range(1, self.N + 1):  # noqa: E741
            for m in range(0, l + 1):
                potential += (
                    rhol[l]
                    * aBar[l, m]
                    * (self.cBar[l, m] * rE[m] + self.sBar[l, m] * iM[m])
                )

        potential += self.mu / r
        neg_potential = tf.negative(potential)
        return neg_potential

    def call(self, inputs):
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        s = inputs_transpose[1]
        t = inputs_transpose[2]
        u = inputs_transpose[3]
        a = tf.ones_like(r) * self.a

        # potential = self.compute_potential(r, s, t, u, a)

        potential = tf.map_fn(
            lambda x: self.compute_potential(x[0], x[1], x[2], x[3], x[4]),
            elems=(r, s, t, u, a),
            fn_output_signature=(r.dtype),
            # parallel_iterations=10
        )
        u = tf.reshape(potential, (-1, 1))
        return u

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "mu": self.mu,
                "a": self.a,
                "n1": self.n1,
                "n2": self.n2,
                "cBar": self.cBar,
                "sBar": self.sBar,
                "N": self.N,
                "dtype": self.dtype,
            },
        )
        return config


class PointMassLayer(tf.keras.layers.Layer):
    def __init__(self, dtype, mu, r_max):
        super(PointMassLayer, self).__init__(dtype=dtype)
        self.mu = tf.constant(mu, dtype=dtype).numpy()

    def call(self, inputs):
        r = tf.linalg.norm(inputs, axis=1, keepdims=True)
        u_pm = tf.negative(tf.divide(self.mu, r))
        return u_pm

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "mu": self.mu,
            },
        )
        return config


class NormalizeRLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        dtype,
        ref_radius_min,
        ref_radius_max,
        feature_min,
        feature_max,
        **kwargs,
    ):
        super(NormalizeRLayer, self).__init__(dtype=dtype)

        self.ref_radius_min = tf.constant(ref_radius_min, dtype=dtype).numpy()
        self.ref_radius_max = tf.constant(ref_radius_max, dtype=dtype).numpy()

        # bounds of the feature -- shouldn't necessarily be a large difference.
        # think about how much the output should change with respect to the inputs
        self.feature_min = tf.constant(feature_min, dtype=dtype).numpy()
        self.feature_max = tf.constant(feature_max, dtype=dtype).numpy()

    def call(self, inputs):
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        s = inputs_transpose[1]
        t = inputs_transpose[2]
        u = inputs_transpose[3]

        df = self.feature_max - self.feature_min
        dr = self.ref_radius_max - self.ref_radius_min
        scale = tf.divide(df, dr)
        min_arg = self.feature_min - tf.multiply(self.ref_radius_min, scale)
        feature = tf.multiply(r, scale) + min_arg

        spheres = tf.stack([feature, s, t, u], axis=1)
        return spheres

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "dtype": self.dtype,
                "feature_min": self.feature_min,
                "feature_max": self.feature_max,
                "ref_radius_min": self.ref_radius_min,
                "ref_radius_max": self.ref_radius_max,
            },
        )
        return config


class ScaleRLayer(tf.keras.layers.Layer):
    def __init__(self, dtype, ref_radius_max, **kwargs):
        super(ScaleRLayer, self).__init__(dtype=dtype)
        self.ref_radius_max = tf.constant(ref_radius_max, dtype=dtype)

    def call(self, inputs):
        inputs_transpose = tf.transpose(inputs)
        r = inputs_transpose[0]
        s = inputs_transpose[1]
        t = inputs_transpose[2]
        u = inputs_transpose[3]
        r_star = tf.divide(r, self.ref_radius_max)
        spheres = tf.stack([r_star, s, t, u], axis=1)
        return spheres

    def get_config(self):
        config = super().get_config().copy()
        config.update({"ref_radius_max": self.ref_radius_max})
        return config


if __name__ == "__main__":
    inputs = np.array([[100.0, -0.1, 0.5, -0.9], [200, 0.2, -0.4, 0.8]])
    layer = FourierFeatureLayer(16, 2)
    layer(inputs)
