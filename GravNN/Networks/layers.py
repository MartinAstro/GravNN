class ResnetIdentityBlock3(tf.keras.Model):
  def __init__(self, nodes, activation):
    super(ResnetIdentityBlock, self).__init__(name='')

    self.dense1a = tf.keras.layers.Dense(nodes)
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.dense2 = tf.keras.layers.Dense(nodes)
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.dense3 = tf.keras.layers.Dense(nodes)
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    act = tf.nn.tanh if activation == 'tanh' else tf.nn.relu

    x = self.dense1(input_tensor)
    x = self.bn2a(x, training=training)
    x = act(x)

    x = self.dense2(x)
    x = self.bn2b(x, training=training)
    x = act(x)

    x = self.dense3(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return act(x)

class DenseBlock(tf.keras.Layer):
    def __init__(self, nodes, num_layers, activation):
