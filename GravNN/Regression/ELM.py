import numpy as np
import tensorflow as tf


# Inspired from:
# https://github.com/burnpiro/elm-pure
# https://github.com/otenim/TensorFlow-OS-ELM/blob/master/train_mnist.py
class OS_ELM:
    def __init__(self, n_input_nodes, n_hidden_nodes, n_output_nodes):
        self.n_input_nodes = n_input_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.n_output_nodes = n_output_nodes
        self.activation = tf.nn.tanh
        self.lossfun = tf.losses.mean_squared_error

        self.w = tf.Variable(
            tf.random.uniform(
                [self.n_input_nodes, self.n_hidden_nodes],
                -1,
                1,
                seed=1234,
            ),
            name="w",
            trainable=False,
        )
        self.beta = tf.Variable(
            tf.random.uniform(
                [self.n_hidden_nodes, self.n_output_nodes],
                -1,
                1,
                seed=1234,
            ),
            name="beta",
            trainable=False,
        )
        self.bias = tf.Variable(
            tf.random.uniform([self.n_hidden_nodes], -1, 1, seed=1234),
            name="bias",
            trainable=False,
        )
        self.P = tf.Variable(
            tf.zeros([self.n_hidden_nodes, self.n_hidden_nodes]),
            name="p",
            trainable=False,
        )

    def predict(self, x):
        H = self.activation(x @ self.w + self.bias)
        y = H @ self.beta
        return y

    def init_train(self, x, y):
        # * is element wise
        # @ is matrix wise

        # Works
        H = self.activation(x @ self.w + self.bias)
        H_pinv = tf.linalg.pinv(H)
        self.beta.assign(H_pinv @ y)

        # Doesn't
        HT = tf.transpose(H)
        HTH_inv = tf.linalg.inv(HT @ H)
        H_inv2 = HTH_inv @ HT
        self.P.assign(HTH_inv)
        self.beta.assign(H_inv2 @ y)

        # HT = tf.transpose(H)
        # # self.P.assign(tf.linalg.pinv(H))
        # self.P.assign(tf.linalg.inv(HT @ H))
        # PHT = self.P @ HT
        # self.beta.assign(PHT @ y)
        return

    def seq_train(self, x, y):
        H = self.activation(x @ self.w + self.bias)
        HT = tf.transpose(H)
        batch_size = tf.shape(x)[0]
        I = tf.eye(batch_size)
        Hp = tf.matmul(H, self.P)
        HpHT = tf.matmul(Hp, HT)
        temp = tf.linalg.inv(I + HpHT)
        PHT = tf.matmul(self.P, HT)
        self.P.assign(self.P - tf.matmul(tf.matmul(PHT, temp), Hp))
        PHT = tf.matmul(self.P, HT)
        Hbeta = tf.matmul(H, self.beta)
        self.beta.assign(self.beta + tf.matmul(PHT, y - Hbeta))
        return


def main():
    np.random.seed(1234)

    n_input_nodes = 1
    n_hidden_nodes = 100
    n_output_nodes = 1

    N = 1000
    N_test = 100
    init_batch = N

    os_elm = OS_ELM(
        n_input_nodes=n_input_nodes,
        n_hidden_nodes=n_hidden_nodes,
        n_output_nodes=n_output_nodes,
    )

    def fcn(x):
        # return np.sin(5 * x)
        return x**2
        return x

    def noisy_fcn(x):
        return fcn(x) + 0.01 * np.random.rand(*x.shape)

    x = np.random.rand(N, n_input_nodes).astype(np.float32)
    y = fcn(x)

    x_test = np.random.rand(N_test, n_input_nodes).astype(np.float32)
    y_test = fcn(x_test)

    x_init = x[:init_batch]
    y_init = y[:init_batch]
    os_elm.init_train(x_init, y_init)

    y_hat = os_elm.predict(x_test)
    L = np.mean(np.square(y_test - y_hat))
    print(L)

    # # iterate through remainder of the vectors in batches of 10
    # step = 1000
    # for i in range(init_batch, len(x), step):
    #     x_seq = x[i:i+step]
    #     y_seq = y[i:i+step]
    #     os_elm.seq_train(x_seq, y_seq)

    # # 'predict' method returns raw values of output nodes.
    # y_hat = os_elm.predict(x_test)
    # L = np.mean(np.square(y_test - y_hat))
    # print(L)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(x, y, s=2)
    plt.scatter(x, os_elm.predict(x), s=2)

    # plt.figure()
    # plt.scatter(x_test, y_test, s=2)
    # plt.scatter(x_test, os_elm.predict(x_test), s=2)
    plt.show()


if __name__ == "__main__":
    main()
