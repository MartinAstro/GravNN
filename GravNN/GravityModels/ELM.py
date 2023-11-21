import os
import pickle

import numpy as np

from GravNN.GravityModels.GravityModelBase import GravityModelBase

np.random.seed(1234)


class ELM(GravityModelBase):
    def __init__(self, filename):
        self.load(filename)
        self.filename = filename
        self.n_input_nodes = self.w.shape[1]
        self.n_hidden_nodes = self.w.shape[0]
        self.n_output_nodes = self.beta.shape[0]
        super().__init__(
            filename,
            self.n_input_nodes,
            self.n_hidden_nodes,
            self.n_output_nodes,
        )

    def generate_full_file_directory(self):
        model_id = os.path.splitext(os.path.basename(self.filename))[0]
        self.file_directory += f"{model_id}/"

    def load(self, name):
        with open(name, "rb") as f:
            params = pickle.load(f)
            self.input_scaler = pickle.load(f)
            self.output_scaler = pickle.load(f)
        self.w = params["w"]
        self.bias = params["bias"]
        self.beta = params["beta"]

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_acceleration(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.shape[0] != self.n_input_nodes:
            x = x.T

        x_nd = self.input_scaler.transform(x.T).T

        H = self.activation(self.w @ x_nd + self.bias)
        y_nd = self.beta @ H

        y = self.output_scaler.inverse_transform(y_nd.T)

        if y.shape[1] != self.n_output_nodes:
            y = y.T

        return y

    def compute_potential(self, x):
        return np.zeros((len(x), 1)) * np.nan


def main():
    n_input_nodes = 1
    N = 5000
    N_test = 100

    elm = ELM(filename)

    def fcn(x):
        return x**2

    x = np.random.uniform(size=(N, n_input_nodes))
    y = fcn(x)

    x_test = np.random.uniform(size=(N_test, n_input_nodes))
    y_test = fcn(x_test)

    y_hat = elm.predict(x_test.T)
    L = np.mean(np.square(y_test - y_hat))
    print(L)

    # 'predict' method returns raw values of output nodes.
    y_hat = elm.predict(x_test)
    L = np.mean(np.square(y_test - y_hat))
    print(L)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(x, y, s=2)
    plt.scatter(x, elm.predict(x.T), s=2)
    plt.show()


if __name__ == "__main__":
    main()
