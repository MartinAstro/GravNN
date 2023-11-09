"""Custom tensorflow callbacks"""
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

# from sigfig import


class SimpleCallback(tf.keras.callbacks.Callback):
    """Simple Callback that prints out loss metrics every 10 epochs and
    measures the amount of time per iteration and total training time"""

    def __init__(self, batch_size, print_interval=10):
        super().__init__()
        self.batch_size = batch_size
        self.print_interval = print_interval
        self.N_train_batches = 0
        self.N_test_batches = 0

        self.N_train_samples = 0
        self.epoch_loss = 0.0
        self.epoch_percent_mean = 0.0
        self.epoch_percent_max = 0.0

        self.N_val_samples = 0
        self.epoch_val_loss = 0.0
        self.epoch_val_percent_mean = 0.0
        self.epoch_val_percent_max = 0.0

    def incremental_average(self, avg, val, N, M):
        return (val * M + avg * N) / (N + M)

    def on_train_batch_end(self, batch, logs=None):
        self.epoch_loss += logs["loss"]
        self.epoch_percent_mean = self.incremental_average(
            self.epoch_percent_mean,
            logs["percent_mean"],
            self.N_train_samples,
            self.batch_size,
        )
        self.N_train_samples += self.batch_size
        self.epoch_percent_max = np.max([logs["percent_max"], self.epoch_percent_max])
        self.N_train_batches += 1

    def on_test_batch_end(self, batch, logs=None):
        # 'val_' prefix is not yet appended until after epoch
        self.epoch_val_loss += logs["loss"]
        self.epoch_val_percent_mean = self.incremental_average(
            self.epoch_val_percent_mean,
            logs["percent_mean"],
            self.N_val_samples,
            self.batch_size,
        )
        self.N_val_samples += self.batch_size
        self.epoch_val_percent_max = np.max(
            [logs["percent_max"], self.epoch_val_percent_max],
        )
        self.N_test_batches += 1

    def on_epoch_begin(self, epoch, logs=None):
        self.N_train_samples = 0
        self.epoch_loss = 0.0
        self.epoch_percent_mean = 0.0
        self.epoch_percent_max = 0.0

        self.N_val_samples = 0
        self.epoch_val_loss = 0.0
        self.epoch_val_percent_mean = 0.0
        self.epoch_val_percent_max = 0.0

        self.N_train_batches = 0
        self.N_test_batches = 0

    def on_epoch_end(self, epoch, logs=None):
        self.N_train_batches = np.max([self.N_train_batches, 1])
        self.N_test_batches = np.max([self.N_test_batches, 1])
        if epoch % self.print_interval == 0:
            print(
                "Epoch: {} \t Loss: {:.9f} \t Val Loss: {:.9f} \t Time: {:.3f} \t \
                    Avg Error: {:.9f}% \t Max Error: {:.9f}%".format(
                    epoch,
                    self.epoch_loss / self.N_train_batches,
                    self.epoch_val_loss / self.N_test_batches,
                    time.time() - self.start_time,
                    self.epoch_val_percent_mean * 100.0,
                    self.epoch_val_percent_max * 100.0,
                ),
            )
            self.start_time = time.time()

        # Overwrite batch logs for epoch logs (to be saved in history obj)
        logs["loss"] = self.epoch_loss / self.N_train_batches
        logs["percent_mean"] = self.epoch_percent_mean
        logs["percent_max"] = self.epoch_percent_max

        logs["val_loss"] = self.epoch_val_loss / self.N_test_batches
        logs["val_percent_mean"] = self.epoch_val_percent_mean
        logs["val_percent_max"] = self.epoch_val_percent_max

    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        self.time_delta = np.round(self.end_time - self.train_start, 2)


class GradientCallback(tf.keras.callbacks.Callback):
    """
    Callback that plots out the gradients for each hidden layer after every 1000
    epochs
    """

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print(
                "Epoch: {} \t Loss: {:.9f} \t Val Loss {:.9f} \t Time: {:.3f}".format(
                    epoch,
                    logs["loss"],
                    logs["val_loss"],
                    time.time() - self.start_time,
                ),
            )
            self.start_time = time.time()
        if epoch % 1000 == 0:
            print("Epoch: {} \t adaptive: {}".format(epoch, logs["adaptive_constant"]))
            fig, ax = plt.subplots()
            np.array([])

            i = 0
            j = 0
            colors = ["b", "r", "g", "y"]

            grad_comps = logs["grads"]
            num_layers = len(grad_comps[0])

            # new figure for each layer
            for i in range(0, num_layers):
                plt.subplot(num_layers // 2, 2, i + 1)

                # plot hist for each gradient type
                for j in range(len(grad_comps)):
                    sns.histplot(
                        grad_comps[j][i].reshape((-1)),
                        ax=plt.gca(),
                        kde=False,
                        label="layer (even), bias(odd) :" + str(i),
                        color=colors[j],
                        binrange=[-0.5, 0.5],
                    )

            plt.legend()

    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        self.time_delta = np.round(self.end_time - self.train_start, 2)


class TimingCallback(tf.keras.callbacks.Callback):
    """Callback that only estimates the amount of time to train the network"""

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            delta = time.time() - self.start_time
            print(
                "Epoch: {} \t Loss: {:.9f} \t Val Loss {:.9f} \t Time: {:.3f}".format(
                    epoch,
                    logs["loss"],
                    logs["val_loss"],
                    delta,
                ),
            )
            self.start_time = time.time()
            self.time_10 = delta

    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        self.time_delta = np.round(self.end_time - self.train_start, 2)


def get_early_stop(config):
    if config["early_stop"][0]:
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-6,
            patience=500,
            verbose=1,
            baseline=None,
            restore_best_weights=True,
        )
