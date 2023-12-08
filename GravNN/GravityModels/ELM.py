import os
import pickle
import tempfile

import numpy as np

from GravNN.GravityModels.GravityModelBase import GravityModelBase
from GravNN.Regression.ELMRegressor import OS_ELM

np.random.seed(1234)


class ELM(GravityModelBase):
    def __init__(self, filename):
        self.load(filename)
        self.filename = filename
        self.n_input_nodes = self.w.shape[1]
        self.n_hidden_nodes = self.w.shape[0]
        self.n_output_nodes = self.beta.shape[0]
        self.max_pred_batch = 500  # ~50 Mb
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
        def pred_mini_batch(x_i):
            if x_i.ndim == 1:
                x_i = x_i.reshape(1, -1)
            if x_i.shape[0] != self.n_input_nodes:
                x_i = x_i.T

            x_nd = self.input_scaler.transform(x_i.T).T

            H = self.activation(self.w @ x_nd + self.bias)
            y_nd = self.beta @ H

            y_i = self.output_scaler.inverse_transform(y_nd.T).T
            return y_i

        N_samples = np.max(x.shape)
        if N_samples > self.max_pred_batch:
            y = np.zeros((N_samples, self.n_output_nodes))
            for i in range(0, N_samples, self.max_pred_batch):
                end_idx = min(i + self.max_pred_batch, N_samples)
                x_batch = x[i:end_idx, :]
                y_batch = pred_mini_batch(x_batch)
                y[i:end_idx, :] = y_batch.T  # (batch, 3)
            y = y.T  # (3, N)
        else:
            y = pred_mini_batch(x)  # (3, N)

        return y.T  # (N, 3)

    def compute_potential(self, x):
        return np.zeros((len(x), 1)) * np.nan


def main():
    n_input_nodes = 1
    N = 5000
    N_test = 100

    n_input_nodes = 1
    n_hidden_nodes = 1000
    n_output_nodes = 1

    N = 50000
    N_test = 10000

    # Regularization Factor C = 5*10**5
    # L = 5*10**4

    # C = 1/k, therefore k = 2E-6

    def fcn(x):
        return x**2

    x = np.random.uniform(size=(N, n_input_nodes))
    y = fcn(x)

    x_test = np.random.uniform(size=(N_test, n_input_nodes))
    y_test = fcn(x_test)

    os_elm = OS_ELM(
        n_input_nodes=n_input_nodes,
        n_hidden_nodes=n_hidden_nodes,
        n_output_nodes=n_output_nodes,
        k=2e-6,
    )
    os_elm.update(x, y, init_batch=1000)

    # Make a temporary file to save the ELM data to
    filename = tempfile.NamedTemporaryFile().name
    os_elm.save(filename)

    elm = ELM(filename)

    y_hat = elm.compute_acceleration(x_test)
    L = np.mean(np.square(y_test - y_hat))
    print(L)

    # 'predict' method returns raw values of output nodes.
    y_hat = elm.compute_acceleration(x_test)
    L = np.mean(np.square(y_test - y_hat))
    print(L)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(x, y, s=2)
    plt.scatter(x, elm.compute_acceleration(x), s=2)
    plt.show()


if __name__ == "__main__":
    main()
