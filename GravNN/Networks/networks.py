import tensorflow as tf

def CustomNet(**kwargs):
    layers = kwargs['layers'][0]
    activation = kwargs['activation'][0]
    dropout_list = kwargs['dropout'][0]
    initializer = kwargs['initializer'][0]
    inputs = tf.keras.Input(shape=(layers[0],))
    x = inputs
    for i in range(1,len(layers)-1):
        x = tf.keras.layers.Dense(units=layers[i], 
                                    activation=activation, 
                                    kernel_initializer=initializer,
                                    )(x)
                                    #dtype=kwargs['dtype'])(x)
        # dropout layers
        if dropout_list[i] != 0.0:
            x = tf.keras.layers.Dropout(dropout_list[i])(x)

    outputs = tf.keras.layers.Dense(units=layers[-1], 
                                    activation='linear', 
                                    kernel_initializer=initializer,
                                    dtype='float32'
                                    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def TraditionalNet(**kwargs):
    layers = kwargs['layers'][0]
    activation = kwargs['activation'][0]
    initializer = kwargs['initializer'][0]

    inputs = tf.keras.Input(shape=(layers[0],))
    x = inputs
    for i in range(1,len(layers)-1):
        x = tf.keras.layers.Dense(units=layers[i], 
                                    activation=activation, 
                                    kernel_initializer=initializer,
                                    )(x)
                                    #dtype=kwargs['dtype'])(x)
        if 'dropout' in kwargs:
            if kwargs['dropout'][0] != 0.0:
                x = tf.keras.layers.Dropout(kwargs['dropout'][0])(x)
    outputs = tf.keras.layers.Dense(units=layers[-1], 
                                    activation='linear', 
                                    kernel_initializer=initializer,
                                    dtype='float32'
                                    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def ResNet(**kwargs):
    layers = kwargs['layers'][0]
    activation = kwargs['activation'][0]
    initializer = kwargs['initializer'][0]

    skip_offset = 3
    inputs = tf.keras.Input(shape=(layers[0],))
    x = inputs
    for i in range(1,len(layers)-1):
        x = tf.keras.layers.Dense(units=layers[i], 
                                    activation=activation, 
                                    kernel_initializer=initializer)(x)
        if (i-1) % skip_offset == 0 and (i-1) == 0:
            skip = x 
        if (i-1) % skip_offset == 0 and (i-1) != 0:
            x = tf.keras.layers.Add()([x, skip])
            x = tf.keras.layers.Activation(activation)(x)
            skip = x 
    outputs = tf.keras.layers.Dense(units=layers[-1], 
                                    activation='linear', 
                                    dtype='float32',
                                    kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def InceptionNet(layers, activation, **kwargs):
    inputs = tf.keras.Input(shape=(layers[0],))
    initializer = kwargs['initializer'][0]

    x = inputs
    for i in range(1,len(layers)-1):
        tensors = []
        for j in range(0, len(layers[i])):
            x_j = tf.keras.layers.Dense(units=layers[i][j], 
                                        activation=activation, 
                                        kernel_initializer=initializer)(x)
            tensors.append(x_j)
        x = tf.keras.layers.Concatenate(axis=1)(tensors)
        x = tf.keras.layers.Activation(activation)(x)
        
    outputs = tf.keras.layers.Dense(units=layers[-1], 
                                    activation='linear', 
                                    kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def DenseNet(layers, activation, **kwargs):
    inputs = tf.keras.Input(shape=(layers[0],))
    initializer = kwargs['initializer'][0]

    x = inputs
    for i in range(1,len(layers)-1):
        tensors = []
        tensors.append(x)
        if len(layers[i]) > 1:
            for j in range(0, len(layers[i])):
                y = tf.keras.layers.Dense(units=layers[i][j], 
                                            activation=activation, 
                                            kernel_initializer=initializer)(x)
                tensors.append(y)
                x = tf.keras.layers.Concatenate(axis=1)(tensors)
        else:
            x = tf.keras.layers.Dense(units=layers[i][0],
                                        activation=activation,
                                        kernel_initializer=initializer)(x)

    outputs = tf.keras.layers.Dense(units=layers[-1], 
                                    activation='linear', 
                                    kernel_initializer=initializer)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model