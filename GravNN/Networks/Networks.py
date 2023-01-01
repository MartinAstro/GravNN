import os
import tensorflow as tf
from GravNN.Networks import utils
from GravNN.Networks.Layers import *
def load_network(config):
    if config["init_file"][0] is not None:
        network = tf.keras.models.load_model(
            os.path.abspath(".")
            + "/Data/Networks/"
            + str(config["init_file"][0])
            + "/network"
        )
    else:
        network_fcn = utils._get_network_fcn(config["network_type"][0])
        network = network_fcn(**config)
    return network


def get_network_fcn(network_type):
    return {
        "traditional" : traditional_network,
        "transformer" : transformer_network,
    }[network_type.lower()]

def get_preprocess_layer_fcn(layer_key):
    return {
        "pines" : Cart2PinesSphLayer,
        "r_scale" : ScaleRLayer,
        "r_normalize" : NormalizeRLayer,
        "r_inv" : InvRLayer,
        "fourier" : FourierFeatureLayer
    }[layer_key.lower()]

def get_preprocess_args(config):
    ref_radius_max = config.get('ref_radius_max', [1E-3])[0]
    ref_radius_min = config.get('ref_radius_min', [1.0])[0]
    feature_min = config.get('feature_min', [1.0])[0]
    feature_max = config.get('feature_max', 
        [1.0 + (ref_radius_max - ref_radius_min)/ref_radius_max])[0]
    preprocess_args = {
        "dtype" : config["dtype"][0],
        "ref_radius_max" : ref_radius_max, 
        "ref_radius_min" : ref_radius_min, 
        "feature_min" : feature_min, 
        "feature_max" : feature_max, 
        "fourier_features" : config.get('fourier_features', [1])[0],
        "fourier_sigma" : config.get('fourier_sigma', [1])[0],
        "fourier_scale" : config.get('fourier_scale', [1])[0],
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
    final_layer_initializer = kwargs.get("final_layer_initializer", ['glorot_uniform'])[0]
    dtype = kwargs["dtype"][0]

    x = inputs
    for i in range(1, len(layers) - 1):
        x = tf.keras.layers.Dense(
            units=layers[i],
            activation=activation,
            kernel_initializer=initializer,
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
        kernel_initializer=final_layer_initializer,
        dtype=dtype,
    )(x)
    return outputs

def transformer_network(inputs, **kwargs):
    """Transformer model that takes 4D spherical coordinates as inputs. This architecture was recommended by the
    Wang2020 PINN Gradient Pathologies paper to help expose symmetries and invariances between different layers within the network.
    """
    # adapted from `forward_pass` (~line 242): https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/master/Helmholtz/Helmholtz2D_model_tf.py

    layers = kwargs["layers"][0]
    activation = kwargs["activation"][0]
    initializer = kwargs["initializer"][0]
    final_layer_initializer = kwargs['final_layer_initializer'][0]
    dtype = kwargs["dtype"][0]
    transformer_units = layers[1]

    x = inputs
    encoder_1 = tf.keras.layers.Dense(
        units=transformer_units,
        activation=activation,
        kernel_initializer=initializer,
        dtype=dtype,
    )(x)
    encoder_2 = tf.keras.layers.Dense(
        units=transformer_units,
        activation=activation,
        kernel_initializer=initializer,
        dtype=dtype,
    )(x)
    
    one = tf.constant(1.0, dtype=dtype, shape=(1,transformer_units))
    for i in range(1, len(layers) - 1):
        x = tf.keras.layers.Dense(
            units=layers[i],
            activation=activation,
            kernel_initializer=initializer,
            dtype=dtype,
        )(x)
        UX = tf.keras.layers.Multiply(dtype=dtype)([x, encoder_1])
        one_minus_x = tf.keras.layers.Subtract(dtype=dtype)([one, x])
        VX = tf.keras.layers.Multiply(dtype=dtype)([one_minus_x, encoder_2])

        x = tf.keras.layers.add([UX, VX],dtype=dtype)

        if "batch_norm" in kwargs:
            if kwargs["batch_norm"][0]:
                x = tf.keras.layers.BatchNormalization()(x)
        if "dropout" in kwargs:
            if kwargs["dropout"][0] != 0.0:
                x = tf.keras.layers.Dropout(kwargs["dropout"][0])(x)
    outputs = tf.keras.layers.Dense(
        units=layers[-1],
        activation="linear",
        kernel_initializer=final_layer_initializer,
        dtype=dtype,
    )(x)
    return outputs

def BasicNet(**kwargs):
    layers = kwargs["layers"][0]
    dtype = kwargs["dtype"][0]
    inputs = tf.keras.Input(shape=(layers[0],), dtype=dtype)
    outputs = get_network_fcn(kwargs['network_arch'][0])(inputs, **kwargs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    super(tf.keras.Model, model).__init__(dtype=dtype)
    return model

def CustomNet(**kwargs):
    layers = kwargs["layers"][0]
    dtype = kwargs["dtype"][0]

    preprocess_args = get_preprocess_args(kwargs)
    preprocess_layers = get_preprocess_layers(kwargs)

    inputs = tf.keras.Input(shape=(layers[0],),dtype=dtype)
    x = inputs   
    for layer in preprocess_layers:
        x = layer(**preprocess_args)(x)
        if layer.__name__ == "Cart2PinesSphLayer":
            features = x 

    u_nn = get_network_fcn(kwargs['network_arch'][0])(x, **kwargs)

    if kwargs.get("scale_nn_potential", [False])[0]:
        u_nn = ScalePotentialNN(**kwargs)(features, u_nn)

    if kwargs.get('deg_removed', [-1])[0] == -1 and kwargs.get('blend_potential', [False])[0]:
        C20 = kwargs.get("cBar",np.zeros((3,3)))[2,0]

        mu = kwargs.get('mu_non_dim', [1.0])[0]
        radius = kwargs['planet'][0].radius
        x_transformer = kwargs['x_transformer'][0]
        radius_non_dim = x_transformer.transform(np.array([[radius, 0,0]]))[0,0]
        ref_radius_analytic = kwargs.get('ref_radius_analytic', [None])[0]
        u_analytic = PlanetaryOblatenessLayer(dtype, mu, radius_non_dim, C20)(features)
        u = BlendPotentialLayer(dtype, mu, ref_radius_analytic)(u_nn, u_analytic, features)
    else:
        u = u_nn

    model = tf.keras.Model(inputs=inputs, outputs=u)
    super(tf.keras.Model, model).__init__(dtype=dtype)

    return model

def MultiScaleNet(**kwargs):
    layers = kwargs["layers"][0]
    dtype = kwargs["dtype"][0]
    fourier_features = kwargs['fourier_features'][0]

    preprocess_args = get_preprocess_args(kwargs)
    preprocess_layers = get_preprocess_layers(kwargs)


    inputs = tf.keras.Input(shape=(layers[0],),dtype=dtype)
    x = inputs   
    for layer in preprocess_layers:
        x = layer(**preprocess_args)(x)
        if layer.__name__ == "Cart2PinesSphLayer":
            features = x 

    fourier_feature_layers = []
    for sigma in kwargs['fourier_sigma'][0]:
        # make a unique fourier feature
        num_features = kwargs['fourier_features'][0]
        freq_decay = kwargs['freq_decay'][0]
        ff_layer = FourierFeatureLayer(num_features, sigma, 1, freq_decay)(x)
        fourier_feature_layers.append(ff_layer)

    sub_net_inputs = tf.keras.Input(shape=(fourier_features+1))
    sub_net = get_network_fcn(kwargs['network_arch'][0])(sub_net_inputs, **kwargs)
    sub_model = tf.keras.Model(inputs=sub_net_inputs, outputs=sub_net)

    u_nn_outputs = []
    for fourier_feature in fourier_feature_layers:
        u_nn_ff = sub_model(fourier_feature)
        u_nn_outputs.append(u_nn_ff)
    
    u_inputs = tf.concat(u_nn_outputs,1)
    u_nn = tf.keras.layers.Dense(1, activation='linear', kernel_initializer='glorot_uniform')(u_inputs)
    
    if kwargs.get("scale_nn_potential", [False])[0]:
        u_nn = ScalePotentialNN(**kwargs)(features, u_nn)

    if kwargs.get('deg_removed', [-1])[0] == -1 and kwargs.get('blend_potential', [False])[0]:
        cBar = kwargs.get("cBar",[0])[0]
        C20 = cBar[2,0]

        mu = kwargs.get('mu_non_dim', [1.0])[0]
        radius = kwargs['planet'][0].radius
        x_transformer = kwargs['x_transformer'][0]
        radius_non_dim = x_transformer.transform(np.array([[radius, 0,0]]))[0,0]
        ref_radius_analytic = kwargs.get('ref_radius_analytic', [None])[0]
        u_analytic = PlanetaryOblatenessLayer(dtype, mu, radius_non_dim, C20)(features)
        u = BlendPotentialLayer(dtype, mu, ref_radius_analytic)(u_nn, u_analytic, features)
    else:
        u = u_nn

    model = tf.keras.Model(inputs=inputs, outputs=u)
    super(tf.keras.Model, model).__init__(dtype=dtype)

    return model


