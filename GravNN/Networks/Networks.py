import os

import tensorflow as tf

from GravNN.Networks.Layers import *


def _get_network_fcn(name):
    return {
        "basic": BasicNet,
        "custom": CustomNet,
        "multi": MultiScaleNet,
        "separation": SeparationNet,
    }[name.lower()]


def load_network(config):
    if config["init_file"][0] is not None:
        network = tf.keras.models.load_model(
            os.path.abspath(".")
            + "/Data/Networks/"
            + str(config["init_file"][0])
            + "/network",
        )
        print(f"LOG: Loaded network from file: {config['init_file'][0]}")
    else:
        network_fcn = _get_network_fcn(config["network_type"][0])
        network = network_fcn(**config)
    return network


def compute_p(**kwargs):
    # define the power of the analytic model
    # if deg_removed == -1: power = 1
    # if np.any(deg_removed == [0,1]): power = 3
    # elif power = deg_removed + 2

    fuse_models = kwargs.get("fuse_models")[0]
    scale_potential = kwargs.get("scale_nn_potential")[0]

    if not scale_potential:
        return 0  # don't scale, regardless of circumstance
    else:
        # at minimum, assume no analytic model and scale using p=1
        p = 1

        # if there is an analytic model, and it is fused with the
        # nn solution, then look for how many terms are used in it
        if fuse_models:
            mu = kwargs.get("mu_non_dim", [0.0])[0]
            C20 = kwargs.get("cBar", [np.zeros((3, 3))])[0][2, 0]

            p += 1 if mu != 0.0 else 0  # if mu is known (for planets and asteroids)
            p += 2 if C20 != 0.0 else 0  # if C20 is known (for planets, not asteroids)

        # if there is no analytic model, then look if any degrees are
        elif not fuse_models and kwargs.get("deg_removed")[0] != -1:
            lr = kwargs.get("deg_removed")[0]
            p = lr + 2

    return p


def get_network_fcn(network_type):
    return {
        "traditional": traditional_network,
        "residual": residual_network,
        "transformer": transformer_network,
        "transformer_siren": transformer_network_siren,
    }[network_type.lower()]


def get_initalizer_fcn(network_type, seed):
    return {
        "glorot_uniform": tf.keras.initializers.GlorotUniform(seed=seed),
        "glorot_normal": tf.keras.initializers.GlorotNormal(seed=seed),
        "zeros": tf.keras.initializers.Zeros(),
    }[network_type.lower()]


def get_preprocess_args(config):
    ref_radius_max = config.get("ref_radius_max", [1e-3])[0]
    ref_radius_min = config.get("ref_radius_min", [1.0])[0]
    feature_min = config.get("feature_min", [1.0])[0]
    feature_max = config.get(
        "feature_max",
        [1.0 + (ref_radius_max - ref_radius_min) / ref_radius_max],
    )[0]
    preprocess_args = {
        "dtype": config["dtype"][0],
        "ref_radius_max": ref_radius_max,
        "ref_radius_min": ref_radius_min,
        "feature_min": feature_min,
        "feature_max": feature_max,
        "fourier_features": config.get("fourier_features", [1])[0],
        "fourier_sigma": config.get("fourier_sigma", [1])[0],
        "trainable": config.get("trainable", [True])[0],
        "freq_decay": config.get("freq_decay", [False])[0],
        "shared_freq": config.get("shared_freq", [False])[0],
        "shared_offset": config.get("shared_offset", [False])[0],
        "sine_and_cosine": config.get("sine_and_cosine", [False])[0],
        "base_2_init": config.get("base_2_init", [False])[0],
    }
    return preprocess_args


def get_preprocess_layers(config):
    preprocess_layers = config.get("preprocessing")[0]
    layers = []
    for layer_key in preprocess_layers:
        layers.append(get_preprocess_layer_fcn(layer_key))
    return layers


def traditional_network(inputs, **kwargs):
    """Vanilla densely connected neural network."""
    layers = kwargs["layers"][0]
    activation = kwargs["activation"][0]
    initializer = kwargs["initializer"][0]
    final_layer_initializer = kwargs.get("final_layer_initializer", ["glorot_uniform"])[
        0
    ]
    dtype = kwargs["dtype"][0]
    seed = kwargs["seed"][0]

    x = inputs
    for i in range(1, len(layers) - 1):
        x = tf.keras.layers.Dense(
            units=layers[i],
            activation=activation,
            kernel_initializer=get_initalizer_fcn(initializer, seed + i),
            dtype=dtype,
        )(x)
        if "batch_norm" in kwargs:
            if kwargs["batch_norm"][0]:
                x = tf.keras.layers.BatchNormalization()(x)
        if "dropout" in kwargs:
            if kwargs["dropout"][0] != 0.0:
                x = tf.keras.layers.Dropout(kwargs["dropout"][0])(x)
    outputs = tf.keras.layers.Dense(
        units=layers[-1],
        activation="linear",
        kernel_initializer=get_initalizer_fcn(final_layer_initializer, seed),
        dtype=dtype,
    )(x)
    return outputs


def residual_network(inputs, **kwargs):
    layers = kwargs["layers"][0]
    activation = kwargs["activation"][0]
    initializer = kwargs["initializer"][0]
    final_layer_initializer = kwargs.get("final_layer_initializer", ["glorot_uniform"])[
        0
    ]
    dtype = kwargs["dtype"][0]
    seed = kwargs["seed"][0]

    encoding_layers = kwargs.get("encoding_layers", [2])[0]
    x = inputs
    for i in range(0, encoding_layers):
        x = tf.keras.layers.Dense(
            units=layers[1],  # default number of params
            activation=activation,
            kernel_initializer=get_initalizer_fcn(initializer, seed + i),
            dtype=dtype,
        )(x)
        if "batch_norm" in kwargs:
            if kwargs["batch_norm"][0]:
                x = tf.keras.layers.BatchNormalization()(x)
        if "dropout" in kwargs:
            if kwargs["dropout"][0] != 0.0:
                x = tf.keras.layers.Dropout(kwargs["dropout"][0])(x)

    for i in range(1, len(layers) - 1):
        shortcut = x
        x = tf.keras.layers.Dense(
            units=layers[i],  # default number of params
            activation=activation,
            kernel_initializer=get_initalizer_fcn(initializer, seed + i),
            dtype=dtype,
        )(x)
        if "batch_norm" in kwargs:
            if kwargs["batch_norm"][0]:
                x = tf.keras.layers.BatchNormalization()(x)
        if "dropout" in kwargs:
            if kwargs["dropout"][0] != 0.0:
                x = tf.keras.layers.Dropout(kwargs["dropout"][0])(x)
        # skip connection
        if i % 3 == 0:
            x = x + shortcut

    outputs = tf.keras.layers.Dense(
        units=layers[-1],
        activation="linear",
        kernel_initializer=get_initalizer_fcn(final_layer_initializer, seed),
        dtype=dtype,
    )(x)
    return outputs


def transformer_network(inputs, **kwargs):
    """Transformer model that takes 4D spherical coordinates as inputs.
    This architecture was recommended by the Wang2020 PINN Gradient Pathologies paper
    to help expose symmetries and invariances between different layers within the
    network.
    """
    # adapted from `forward_pass` (~line 242): https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/master/Helmholtz/Helmholtz2D_model_tf.py

    layers = kwargs["layers"][0]
    activation = kwargs["activation"][0]
    initializer = kwargs["initializer"][0]
    final_layer_initializer = kwargs["final_layer_initializer"][0]
    dtype = kwargs["dtype"][0]
    transformer_units = layers[1]
    seed = kwargs["seed"][0]

    x = inputs
    encoder_1 = tf.keras.layers.Dense(
        units=transformer_units,
        activation=activation,
        kernel_initializer=get_initalizer_fcn(initializer, seed + 1234),
        dtype=dtype,
    )(x)
    encoder_2 = tf.keras.layers.Dense(
        units=transformer_units,
        activation=activation,
        kernel_initializer=get_initalizer_fcn(initializer, seed + 12345),
        dtype=dtype,
    )(x)

    one = tf.constant(1.0, dtype=dtype, shape=(1, transformer_units))
    for i in range(1, len(layers) - 1):
        x = tf.keras.layers.Dense(
            units=layers[i],
            activation=activation,
            kernel_initializer=get_initalizer_fcn(initializer, seed + i),
            dtype=dtype,
        )(x)
        UX = tf.keras.layers.Multiply(dtype=dtype)([x, encoder_1])
        one_minus_x = tf.keras.layers.Subtract(dtype=dtype)([one, x])
        VX = tf.keras.layers.Multiply(dtype=dtype)([one_minus_x, encoder_2])

        x = tf.keras.layers.add([UX, VX], dtype=dtype)

        if "batch_norm" in kwargs:
            if kwargs["batch_norm"][0]:
                x = tf.keras.layers.BatchNormalization()(x)
        if "dropout" in kwargs:
            if kwargs["dropout"][0] != 0.0:
                x = tf.keras.layers.Dropout(kwargs["dropout"][0])(x)
    outputs = tf.keras.layers.Dense(
        units=layers[-1],
        activation="linear",
        kernel_initializer=get_initalizer_fcn(final_layer_initializer, seed),
        dtype=dtype,
    )(x)
    return outputs


def transformer_network_siren(inputs, **kwargs):
    """Transformer model that takes 4D spherical coordinates as inputs.
    This architecture was recommended by the Wang2020 PINN Gradient Pathologies paper
    to help expose symmetries and invariances between different layers within the
    network.
    """
    # adapted from `forward_pass` (~line 242): https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/master/Helmholtz/Helmholtz2D_model_tf.py

    layers = kwargs["layers"][0]
    seed = kwargs["seed"][0]
    in_features = 4
    omega_0 = 30.0

    activation = tf.sin
    w_0_initializer = tf.keras.initializers.RandomUniform(
        minval=-1.0 / in_features,
        maxval=1.0 / in_features,
        seed=1234,
    )

    kwargs["final_layer_initializer"][0]
    dtype = kwargs["dtype"][0]
    transformer_units = layers[1]

    omega_layer = tf.constant(omega_0, dtype=dtype, shape=(1, layers[2]))
    omega_layer_inputs = tf.constant(omega_0, dtype=dtype, shape=(1, 64))

    x = inputs
    w_0_initializer = tf.keras.initializers.RandomUniform(
        minval=-1.0 / in_features,
        maxval=1.0 / in_features,
        seed=seed + 1234,
    )
    encoder_1 = tf.keras.layers.Dense(
        units=transformer_units,
        activation=activation,
        kernel_initializer=w_0_initializer,
        dtype=dtype,
    )(x)
    w_0_initializer = tf.keras.initializers.RandomUniform(
        minval=-1.0 / in_features,
        maxval=1.0 / in_features,
        seed=seed + 12345,
    )
    encoder_2 = tf.keras.layers.Dense(
        units=transformer_units,
        activation=activation,
        kernel_initializer=w_0_initializer,
        dtype=dtype,
    )(x)

    one = tf.constant(1.0, dtype=dtype, shape=(1, transformer_units))
    x = tf.keras.layers.multiply([x, omega_layer_inputs])
    for i in range(1, len(layers) - 1):
        w_i_initializer = tf.keras.initializers.RandomUniform(
            minval=-np.sqrt(6 / in_features) / omega_0,
            maxval=np.sqrt(6 / in_features) / omega_0,
            seed=seed + i,
        )

        x = tf.keras.layers.Dense(
            units=layers[i],
            activation=activation,
            kernel_initializer=w_i_initializer,
            dtype=dtype,
        )(x)
        UX = tf.keras.layers.Multiply(dtype=dtype)([x, encoder_1])
        one_minus_x = tf.keras.layers.Subtract(dtype=dtype)([one, x])
        VX = tf.keras.layers.Multiply(dtype=dtype)([one_minus_x, encoder_2])

        x = tf.keras.layers.add([UX, VX], dtype=dtype)
        x = tf.keras.layers.multiply([x, omega_layer])
        if "batch_norm" in kwargs:
            if kwargs["batch_norm"][0]:
                x = tf.keras.layers.BatchNormalization()(x)
        if "dropout" in kwargs:
            if kwargs["dropout"][0] != 0.0:
                x = tf.keras.layers.Dropout(kwargs["dropout"][0])(x)

    w_i_initializer = tf.keras.initializers.RandomUniform(
        minval=-np.sqrt(6 / in_features) / omega_0,
        maxval=np.sqrt(6 / in_features) / omega_0,
        seed=seed + i,
    )
    outputs = tf.keras.layers.Dense(
        units=layers[-1],
        activation="linear",
        kernel_initializer=w_i_initializer,
        dtype=dtype,
    )(x)
    return outputs


def BasicNet(**kwargs):
    layers = kwargs["layers"][0]
    dtype = kwargs["dtype"][0]
    inputs = tf.keras.Input(shape=(layers[0],), dtype=dtype)
    outputs = get_network_fcn(kwargs["network_arch"][0])(inputs, **kwargs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    super(tf.keras.Model, model).__init__(dtype=dtype)
    return model


# Necessary for backwards compatibility
TraditionalNet = BasicNet


def CustomNet(**kwargs):
    layers = kwargs["layers"][0]
    dtype = kwargs["dtype"][0]

    preprocess_args = get_preprocess_args(kwargs)
    preprocess_layers = get_preprocess_layers(kwargs)

    inputs = tf.keras.Input(shape=(layers[0],), dtype=dtype)
    x = inputs
    for layer in preprocess_layers:
        x = layer(**preprocess_args)(x)
        if layer.__name__ == "Cart2PinesSphLayer":
            features = x

    u_nn = get_network_fcn(kwargs["network_arch"][0])(x, **kwargs)

    p = compute_p(**kwargs)
    u_analytic = AnalyticModelLayer(**kwargs)(features)
    u_nn_scaled = ScaleNNPotential(p, **kwargs)(features, u_nn)
    u_fused = FuseModels(**kwargs)(u_nn_scaled, u_analytic)
    u = EnforceBoundaryConditions(**kwargs)(features, u_fused, u_analytic)

    model = tf.keras.Model(inputs=inputs, outputs=u)
    super(tf.keras.Model, model).__init__(dtype=dtype)

    return model


def SeparationNet(**kwargs):
    layers = kwargs["layers"][0]
    dtype = kwargs["dtype"][0]

    preprocess_args = get_preprocess_args(kwargs)
    preprocess_layers = get_preprocess_layers(kwargs)

    inputs = tf.keras.Input(shape=(layers[0],), dtype=dtype)
    x = inputs
    for layer in preprocess_layers:
        x = layer(**preprocess_args)(x)
        if layer.__name__ == "Cart2PinesSphLayer":
            features = x

    # Encode the angles into a network
    # purposefully exclude r coordinate
    kwargs_copy = kwargs.copy()
    kwargs_copy["layers"] = [[-1, 20, 20, 20, 20]]
    angles_nn = get_network_fcn(kwargs["network_arch"][0])(x[:, 1:], **kwargs_copy)

    # fold the r coordinate with the angle nn
    # this acts as seperation of variable
    kwargs_copy["layers"] = [[-1, 20, 20, 20, 1]]
    u_nn_inputs = tf.keras.layers.Concatenate()([x[:, 0:1], angles_nn])
    u_nn = get_network_fcn(kwargs["network_arch"][0])(u_nn_inputs, **kwargs_copy)

    p = compute_p(**kwargs)
    u_analytic = AnalyticModelLayer(**kwargs)(features)
    u_nn_scaled = ScaleNNPotential(p, **kwargs)(features, u_nn)
    u_fused = FuseModels(**kwargs)(u_nn_scaled, u_analytic)
    u = EnforceBoundaryConditions(**kwargs)(features, u_fused, u_analytic)

    model = tf.keras.Model(inputs=inputs, outputs=u)
    super(tf.keras.Model, model).__init__(dtype=dtype)

    return model


def MultiScaleNet(**kwargs):
    layers = kwargs["layers"][0]
    dtype = kwargs["dtype"][0]
    fourier_features = kwargs["fourier_features"][0]

    preprocess_args = get_preprocess_args(kwargs)
    preprocess_layers = get_preprocess_layers(kwargs)

    inputs = tf.keras.Input(shape=(layers[0],), dtype=dtype)
    x = inputs
    for layer in preprocess_layers:
        x = layer(**preprocess_args)(x)
        if layer.__name__ == "Cart2PinesSphLayer":
            features = x

    fourier_feature_layers = []
    for sigma in kwargs["fourier_sigma"][0]:
        # make a unique fourier feature
        num_features = kwargs["fourier_features"][0]
        freq_decay = kwargs["freq_decay"][0]
        ff_layer = FourierFeatureLayer(num_features, sigma, 1, freq_decay)(x)
        fourier_feature_layers.append(ff_layer)

    sub_net_inputs = tf.keras.Input(shape=(fourier_features + 1))
    sub_net = get_network_fcn(kwargs["network_arch"][0])(sub_net_inputs, **kwargs)
    sub_model = tf.keras.Model(inputs=sub_net_inputs, outputs=sub_net)

    u_nn_outputs = []
    for fourier_feature in fourier_feature_layers:
        u_nn_ff = sub_model(fourier_feature)
        u_nn_outputs.append(u_nn_ff)

    u_inputs = tf.concat(u_nn_outputs, 1)
    u_nn = tf.keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer="glorot_uniform",
        dtype=dtype,
    )(u_inputs)

    p = compute_p(**kwargs)
    u_analytic = AnalyticModelLayer(**kwargs)(features)
    u_nn_scaled = ScaleNNPotential(p, **kwargs)(features, u_nn)
    u_fused = FuseModels(**kwargs)(u_nn_scaled, u_analytic)
    u = EnforceBoundaryConditions(**kwargs)(features, u_fused, u_analytic)
    model = tf.keras.Model(inputs=inputs, outputs=u)
    super(tf.keras.Model, model).__init__(dtype=dtype)

    return model


## Legacy


def SphericalTraditionalNet(**kwargs):
    layers = kwargs["layers"][0]
    dtype = kwargs["dtype"][0]

    preprocess_args = get_preprocess_args(kwargs)
    inputs = tf.keras.Input(shape=(layers[0],), dtype=dtype)
    features = Cart2PinesSphLayer(**preprocess_args)(inputs)
    u_nn = get_network_fcn(kwargs["network_arch"][0])(features, **kwargs)
    model = tf.keras.Model(inputs=inputs, outputs=u_nn)
    super(tf.keras.Model, model).__init__(dtype=dtype)

    return model
