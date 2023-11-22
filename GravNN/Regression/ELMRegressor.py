import pickle

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from GravNN.Support.ProgressBar import ProgressBar

np.random.seed(1234)


class OS_ELM:
    def __init__(self, n_input_nodes, n_hidden_nodes, n_output_nodes, k=1):
        self.n_input_nodes = n_input_nodes
        self.n_hidden_nodes = n_hidden_nodes
        self.n_output_nodes = n_output_nodes
        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()

        self.w = np.random.uniform(
            -1,
            1,
            size=(self.n_hidden_nodes, self.n_input_nodes),
        )
        self.bias = np.random.uniform(-1, 1, size=(self.n_hidden_nodes, 1))
        self.beta = np.zeros(shape=(self.n_output_nodes, self.n_hidden_nodes))
        self.c = 1.0 / k
        self.K_inv = np.zeros(shape=(self.n_hidden_nodes, self.n_hidden_nodes))

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[0] != self.n_input_nodes:
            x = x.T

        x_nd = self.input_scaler.transform(x.T).T

        H = self.activation(self.w * x_nd + self.bias)
        y_nd = self.beta @ H

        y = self.output_scaler.inverse_transform(y_nd.T).T
        return y

    def init_train(self, x, y):
        # * is element wise
        # @ is matrix wise

        # X [MxN]
        # W [HxM]
        # H = [HxN]
        # Beta [TxH]
        # Y [TxN]

        # Normalize the inputs and outputs using sklearn
        x = self.input_scaler.fit_transform(x.T).T
        y = self.output_scaler.fit_transform(y.T).T

        H = self.activation(self.w @ x + self.bias)
        Ht = np.transpose(H)
        K = H @ Ht + np.eye((len(H)), dtype=H.dtype) / self.c
        K_inv = np.linalg.inv(K)

        Yt = np.transpose(y)
        beta_t = K_inv @ H @ Yt
        beta = np.transpose(beta_t)

        self.beta = beta
        self.K_inv = K_inv

    def seq_train(self, x, y):
        # Normalize the inputs and outputs using sklearn
        x = self.input_scaler.transform(x.T).T
        y = self.output_scaler.transform(y.T).T

        # If you want to solve for Beta, then transpose everything
        H = self.activation(self.w @ x + self.bias)
        Ht = np.transpose(H)
        Kk_inv = self.K_inv

        eye = np.eye(len(Ht), dtype=H.dtype)

        intermediate = np.linalg.inv(eye + Ht @ Kk_inv @ H)
        dK = Kk_inv @ H @ intermediate @ Ht @ Kk_inv
        new_K_inv = Kk_inv - dK

        # Correct Beta
        y_hat = self.predict(x)
        dy = y - y_hat
        dBeta = new_K_inv @ H @ np.transpose(dy)
        new_beta = self.beta + np.transpose(dBeta)

        self.beta = new_beta
        self.K_inv = new_K_inv

    def update(self, x, y, init_batch):
        if x.shape[0] != self.n_input_nodes:
            x = x.T
        if y.shape[0] != self.n_output_nodes:
            y = y.T
        BS = init_batch
        x_init = x[:, :BS]
        y_init = y[:, :BS]

        self.init_train(x_init, y_init)
        pbar = ProgressBar(len(x[0]), True)
        for i in range(BS, len(x[0]), BS):
            end_idx = min(i + BS, len(x[0]))
            x_batch = x[:, i:end_idx]
            y_batch = y[:, i:end_idx]
            self.seq_train(x_batch, y_batch)
            pbar.update(end_idx)

    def save(self, name):
        params = {
            "w": self.w,
            "bias": self.bias,
            "beta": self.beta,
        }

        with open(name, "wb") as f:
            pickle.dump(params, f)
            pickle.dump(self.input_scaler, f)
            pickle.dump(self.output_scaler, f)


def main():
    n_input_nodes = 1
    n_hidden_nodes = 100
    n_output_nodes = 1

    N = 5000
    N_test = 100
    init_batch = 1000

    # Regularization Factor C = 5*10**5
    # L = 5*10**4

    # C = 1/k, therefore k = 2E-6

    os_elm = OS_ELM(
        n_input_nodes=n_input_nodes,
        n_hidden_nodes=n_hidden_nodes,
        n_output_nodes=n_output_nodes,
        k=2e-6,
    )

    def fcn(x):
        # return np.sin(5 * x)
        return x**2  # + np.sin(5*x)
        return x

    x = np.random.uniform(size=(N, n_input_nodes))
    y = fcn(x)

    x_test = np.random.uniform(size=(N_test, n_input_nodes))
    y_test = fcn(x_test)
    os_elm.update(x.T, y.T, init_batch=init_batch)

    y_hat = os_elm.predict(x_test.T)
    L = np.mean(np.square(y_test - y_hat))
    print(L)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(x, y, s=2)
    plt.scatter(x, os_elm.predict(x.T), s=2)
    plt.show()


if __name__ == "__main__":
    main()
