from GravNN.Networks.Layers import *
import tensorflow as tf

import os
import warnings
from GravNN.Networks import utils
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


def TraditionalNet(**kwargs):
    """Vanilla densely connected neural network.

    TODO: fix keyword acquisition such that some parameters can be optional.

    Args:
        layers (list): list of number of nodes per layer (i.e. [3,10,10,10,3] has 3 inputs nodes, followed by a first
        layer with 10 nodes, followed by a second layer with 10, ...)
        activation (str): non-linear activation function to be used
        initializer (str): weight and bias initialization strategy (ex. 'glorot_normal' or 'glorot_uniform')
        dtype (str) : float dtype (ex. 'float32' or 'float64') -- this is especially important if using mixed precision in TF.
        dropout (float, optional) : fraction of nodes to be dropped between each hidden layer (0.0 means no nodes dropped, 0.5 means half, ...)
    Returns:
        tf.keras.Model: densely connected network
    """
    layers = kwargs["layers"][0]
    activation = kwargs["activation"][0]
    initializer = kwargs["initializer"][0]
    dtype = kwargs["dtype"][0]
    inputs = tf.keras.Input(shape=(layers[0],), dtype=dtype)
    x = inputs
    for i in range(1, len(layers) - 1):
        x = tf.keras.layers.Dense(
            units=layers[i],
            activation=activation,
            kernel_initializer=initializer,
            dtype=dtype,
        )(x)
        if "dropout" in kwargs:
            if kwargs["dropout"][0] != 0.0:
                x = tf.keras.layers.Dropout(kwargs["dropout"][0])(x)
    outputs = tf.keras.layers.Dense(
        units=layers[-1],
        activation="linear",
        kernel_initializer=initializer,
        dtype=dtype,
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    super(tf.keras.Model, model).__init__(dtype=dtype)
    return model

def SphericalPinesTraditionalNet(**kwargs):
    """Densely connected neural network that will convert inputs into 4D spherical coordinates
    before proceeding into the network.

    .. Note:: This network superseeds the SphericalTraditionalNet as its spherical derivatives are non-singular.

    TODO: fix keyword acquisition such that some parameters can be optional.

    Args:
        layers (list): list of number of nodes per layer (i.e. [3,10,10,10,3] has 3 inputs nodes, followed by a first
        layer with 10 nodes, followed by a second layer with 10, ...)
        activation (str): non-linear activation function to be used
        initializer (str): weight and bias initialization strategy (ex. 'glorot_normal' or 'glorot_uniform')
        dtype (str): float dtype (ex. 'float32' or 'float64') -- this is especially important if using mixed precision in TF.
        dropout (float, optional): fraction of nodes to be dropped between each hidden layer (0.0 means no nodes dropped, 0.5 means half, ...)
        custom_input_layer (str): selects any custom configuration option for the layer that enters the network. (e.g. concatenate the cartesian inputs
        with the spherical coordinates using "cart_and_sph").
        skip_normalization (bool): flag determining if the spherical values entering the network should be normalized
        norm_mins (tf.Tensor or np.array): values used to bias the spherical inputs to the network before scaling such that inputs will ultimately be between [-1,1]
        norm_scalers (tf.Tensor or np.array): values used to scale the spherical inputs to the network after biasing such that inputs will be between [-1,1]
        batch_norm (bool, optional): Flag determining if batch normalization layers should be inserted between hidden layers.
    Returns:
        tf.keras.Model: densely connected network
    """
    layers = kwargs["layers"][0]
    activation = kwargs["activation"][0]
    initializer = kwargs["initializer"][0]
    custom_input_layer = kwargs["custom_input_layer"][0]
    dtype = kwargs["dtype"][0]
    skip_normalization = kwargs["skip_normalization"][0]
    ref_radius = kwargs["ref_radius"][0]

    inputs = tf.keras.Input(shape=(layers[0],),dtype=dtype)
    x = Cart2PinesSphLayer(dtype)(inputs)

    if not skip_normalization:
        scalers = kwargs["norm_scalers"][0]
        mins = kwargs["norm_mins"][0]
        x = PinesSph2NetLayer(dtype, scalers, mins, ref_radius)(x)

    if custom_input_layer == "cart_and_sph":
        x = tf.keras.layers.Concatenate(axis=-1)([inputs, x])

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
        kernel_initializer=initializer,
        dtype=dtype,
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    super(tf.keras.Model, model).__init__(dtype=dtype)

    return model

def SphericalPinesTransformerNet(**kwargs):
    """Transformer model that takes 4D spherical coordinates as inputs. This architecture was recommended by the
    Wang2020 PINN Gradient Pathologies paper to help expose symmetries and invariances between different layers within the network.

    TODO: fix keyword acquisition such that some parameters can be optional.
    TODO: Check if the transformer_units have to be the same as the hidden layer node count. If so remove the keyword. s

    Args:
        layers (list): list of number of nodes per layer (i.e. [3,10,10,10,3] has 3 inputs nodes, followed by a first
        layer with 10 nodes, followed by a second layer with 10, ...)
        transformer_units (int): number of nodes used within the encoder layers.
        activation (str): non-linear activation function to be used
        initializer (str): weight and bias initialization strategy (ex. 'glorot_normal' or 'glorot_uniform')
        dtype (str): float dtype (ex. 'float32' or 'float64') -- this is especially important if using mixed precision in TF.
        dropout (float, optional): fraction of nodes to be dropped between each hidden layer (0.0 means no nodes dropped, 0.5 means half, ...)
        skip_normalization (bool): flag determining if the spherical values entering the network should be normalized
        norm_mins (tf.Tensor or np.array): values used to bias the spherical inputs to the network before scaling such that inputs will ultimately be between [-1,1]
        norm_scalers (tf.Tensor or np.array): values used to scale the spherical inputs to the network after biasing such that inputs will be between [-1,1]
        batch_norm (bool, optional): Flag determining if batch normalization layers should be inserted between hidden layers.
    Returns:
        tf.keras.Model: densely connected network
    """
    layers = kwargs["layers"][0]
    activation = kwargs["activation"][0]
    initializer = kwargs["initializer"][0]
    scalers = kwargs["norm_scalers"][0]
    mins = kwargs["norm_mins"][0]
    dtype = kwargs["dtype"][0]
    transformer_units = layers[1]
    normalization_strategy = kwargs["normalization_strategy"][0]
    ref_radius = kwargs["ref_radius"][0]
    inputs = tf.keras.Input(shape=(layers[0],),dtype=dtype)
    x = Cart2PinesSphLayer(dtype)(inputs)


    if normalization_strategy == 'radial':
        x = PinesSph2NetRefLayer(dtype, scalers, mins, ref_radius)(x)
    if normalization_strategy == 'uniform':
        x = PinesSph2NetLayer(dtype, scalers, mins, ref_radius)(x)


    # adapted from `forward_pass` (~line 242): https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/master/Helmholtz/Helmholtz2D_model_tf.py
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
        VX = tf.keras.layers.Multiply(dtype=dtype)([one- x, encoder_2])

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
        kernel_initializer='glorot_uniform',
        dtype=dtype,
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    super(tf.keras.Model, model).__init__(dtype=dtype)

    return model

def SphericalPinesTraditionalNet_v2(**kwargs):
    """Densely connected neural network that will convert inputs into 4D spherical coordinates
    before proceeding into the network.

    .. Note:: This network superseeds the SphericalTraditionalNet as its spherical derivatives are non-singular.

    TODO: fix keyword acquisition such that some parameters can be optional.

    Args:
        layers (list): list of number of nodes per layer (i.e. [3,10,10,10,3] has 3 inputs nodes, followed by a first
        layer with 10 nodes, followed by a second layer with 10, ...)
        activation (str): non-linear activation function to be used
        initializer (str): weight and bias initialization strategy (ex. 'glorot_normal' or 'glorot_uniform')
        dtype (str): float dtype (ex. 'float32' or 'float64') -- this is especially important if using mixed precision in TF.
        dropout (float, optional): fraction of nodes to be dropped between each hidden layer (0.0 means no nodes dropped, 0.5 means half, ...)
        custom_input_layer (str): selects any custom configuration option for the layer that enters the network. (e.g. concatenate the cartesian inputs
        with the spherical coordinates using "cart_and_sph").
        skip_normalization (bool): flag determining if the spherical values entering the network should be normalized
        norm_mins (tf.Tensor or np.array): values used to bias the spherical inputs to the network before scaling such that inputs will ultimately be between [-1,1]
        norm_scalers (tf.Tensor or np.array): values used to scale the spherical inputs to the network after biasing such that inputs will be between [-1,1]
        batch_norm (bool, optional): Flag determining if batch normalization layers should be inserted between hidden layers.
    Returns:
        tf.keras.Model: densely connected network
    """
    layers = kwargs["layers"][0]
    activation = kwargs["activation"][0]
    initializer = kwargs["initializer"][0]
    dtype = kwargs["dtype"][0]

    inputs = tf.keras.Input(shape=(layers[0],), dtype=dtype)
    x = Cart2PinesSphLayer(dtype)(inputs)
    x = PinesSph2NetLayer_v2(dtype)(x)
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
            if kwargs["dropout"][0] != 0.0 and (i != (len(layers) - 2)):
                x = tf.keras.layers.Dropout(kwargs["dropout"][0])(x)
    outputs = tf.keras.layers.Dense(
        units=layers[-1],
        activation="linear",
        kernel_initializer=initializer,
        dtype=dtype,
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    super(tf.keras.Model, model).__init__(dtype=dtype)

    return model

def SphericalPinesTransformerNet_v2(**kwargs):
    """Transformer model that takes 4D spherical coordinates as inputs. This architecture was recommended by the
    Wang2020 PINN Gradient Pathologies paper to help expose symmetries and invariances between different layers within the network.

    TODO: fix keyword acquisition such that some parameters can be optional.
    TODO: Check if the transformer_units have to be the same as the hidden layer node count. If so remove the keyword. s

    Args:
        layers (list): list of number of nodes per layer (i.e. [3,10,10,10,3] has 3 inputs nodes, followed by a first
        layer with 10 nodes, followed by a second layer with 10, ...)
        transformer_units (int): number of nodes used within the encoder layers.
        activation (str): non-linear activation function to be used
    initializer (str): weight and bias initialization strategy (ex. 'glorot_normal' or 'glorot_uniform')
        dtype (str): float dtype (ex. 'float32' or 'float64') -- this is especially important if using mixed precision in TF.
        dropout (float, optional): fraction of nodes to be dropped between each hidden layer (0.0 means no nodes dropped, 0.5 means half, ...)
        skip_normalization (bool): flag determining if the spherical values entering the network should be normalized
        norm_mins (tf.Tensor or np.array): values used to bias the spherical inputs to the network before scaling such that inputs will ultimately be between [-1,1]
        norm_scalers (tf.Tensor or np.array): values used to scale the spherical inputs to the network after biasing such that inputs will be between [-1,1]
        batch_norm (bool, optional): Flag determining if batch normalization layers should be inserted between hidden layers.
    Returns:
        tf.keras.Model: densely connected network
    """
    layers = kwargs["layers"][0]
    activation = kwargs["activation"][0]
    initializer = kwargs["initializer"][0]
    dtype = kwargs.get("dtype", [tf.float32])[0]
    transformer_units = kwargs["num_units"][0]
    ref_radius_max = kwargs.get('ref_radius_max', [None])[0]
    ref_radius_min = kwargs.get('ref_radius_min', [None])[0]


    inputs = tf.keras.Input(shape=(layers[0],), dtype=dtype)
    x = Cart2PinesSphLayer(dtype)(inputs)
    x = PinesSph2NetLayer_v2(dtype, ref_radius_min=ref_radius_min, ref_radius_max=ref_radius_max)(x)

    # adapted from `forward_pass` (~line 242): https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/master/Helmholtz/Helmholtz2D_model_tf.py
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
        V = tf.keras.layers.Subtract(dtype=dtype)([one, x])
        VX = tf.keras.layers.Multiply(dtype=dtype)([V, encoder_2])

        x = tf.keras.layers.add([UX, VX], dtype=dtype)

        if "batch_norm" in kwargs:
            if kwargs["batch_norm"][0]:
                x = tf.keras.layers.BatchNormalization(dtype=dtype)(x)
        if "dropout" in kwargs:
            if kwargs["dropout"][0] != 0.0:
                x = tf.keras.layers.Dropout(kwargs["dropout"][0], dtype=dtype)(x)
    outputs = tf.keras.layers.Dense(
        units=layers[-1],
        activation="linear",
        kernel_initializer='glorot_uniform',
        dtype=dtype,
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    super(tf.keras.Model, model).__init__(dtype=dtype)
    return model



def _heaviside_cond(dr):
    zero = tf.constant(0.0, dtype=dr.dtype)
    one = tf.constant(1.0, dtype=dr.dtype)
    heaviside = tf.cond(dr > zero, lambda: one, lambda : zero)
    return heaviside

def heaviside(dr):
    dr = tf.reshape(dr, [-1,1])
    heaviside = tf.map_fn(fn=lambda dr: _heaviside_cond(dr), elems=dr)
    return heaviside

def SphericalPinesTransformerNet_v3(**kwargs):

    layers = kwargs["layers"][0]
    activation = kwargs["activation"][0]
    initializer = kwargs["initializer"][0]
    dtype = kwargs.get("dtype", [tf.float32])[0]
    transformer_units = kwargs["num_units"][0]
    ref_radius_max = kwargs.get('ref_radius_max', [None])[0]
    ref_radius_min = kwargs.get('ref_radius_min', [None])[0]
    mu = kwargs.get('mu_non_dim', [1.0])[0]

    inputs = tf.keras.Input(shape=(layers[0],), dtype=dtype)
    features = Cart2PinesSphLayer(dtype)(inputs)
    x = PinesSph2NetLayer_v2(dtype, 
                    ref_radius_min=ref_radius_min, 
                    ref_radius_max=ref_radius_max)(features)

    # adapted from `forward_pass` (~line 242): https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/blob/master/Helmholtz/Helmholtz2D_model_tf.py
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
        V = tf.keras.layers.Subtract(dtype=dtype)([one, x])
        VX = tf.keras.layers.Multiply(dtype=dtype)([V, encoder_2])

        x = tf.keras.layers.add([UX, VX], dtype=dtype)

        if "batch_norm" in kwargs:
            if kwargs["batch_norm"][0]:
                x = tf.keras.layers.BatchNormalization(dtype=dtype)(x)
        if "dropout" in kwargs:
            if kwargs["dropout"][0] != 0.0:
                x = tf.keras.layers.Dropout(kwargs["dropout"][0], dtype=dtype)(x)
    u_nn = tf.keras.layers.Dense(
        units=layers[-1],
        activation="linear",
        kernel_initializer='glorot_uniform',
        dtype=dtype,
    )(x)

    u = AugmentedPotentialLayer_v2(dtype, mu, ref_radius_max)(u_nn, inputs)

    model = tf.keras.Model(inputs=inputs, outputs=u)
    super(tf.keras.Model, model).__init__(dtype=dtype)
    return model
