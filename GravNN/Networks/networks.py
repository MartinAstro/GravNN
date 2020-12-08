import tensorflow as tf

def DenseNet(layers, activation, **kwargs):
    inputs = tf.keras.Input(shape=(layers[0],))
    x = inputs
    for i in range(1,len(layers)-1):
        x = tf.keras.layers.Dense(units=layers[i], 
                                    activation=activation, 
                                    kernel_initializer='glorot_normal')(x)
    outputs = tf.keras.layers.Dense(units=layers[-1], 
                                    activation='linear', 
                                    kernel_initializer='glorot_normal')(x)
    #model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return inputs, outputs

def ResNet(layers, activation, **kwargs):
    skip_offset = kwargs['skip_offset']
    inputs = tf.keras.Input(shape=(layers[0],))
    if activation == 'tanh':
        activation = tf.tanh
    elif acitvation == 'relu':
        activation = tf.relu
    else:
        print("Activation not currently compatible with resnet")
        exit(1)
    x = inputs
    skip = inputs
    for i in range(1,len(layers)-1):
        x = tf.keras.layers.Dense(units=layers[i], 
                                    activation=activation, 
                                    kernel_initializer='glorot_normal')(x)
        if i % skip_offset == 0:
            x += skip
            x = activation(x)
            skip = x 
    outputs = tf.keras.layers.Dense(units=layers[-1], 
                                    activation='linear', 
                                    kernel_initializer='glorot_normal')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def InceptionNet(layers, activation, **kwargs):
    inputs = tf.keras.Input(shape=(layers[0],))
    x = inputs
    for i in range(1,len(layers)-1):
        for j in rante(0, len(layers[i])):
            x_j = tf.keras.layers.Dense(units=layers[i], 
                                        activation=activation, 
                                        kernel_initializer='glorot_normal')(x)
            x += x_j
        x = activation(x)                 
    outputs = tf.keras.layers.Dense(units=layers[-1][0], 
                                    activation='linear', 
                                    kernel_initializer='glorot_normal')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model