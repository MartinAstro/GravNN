"""Custom tensorflow callbacks"""
import tensorflow as tf
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class SimpleCallback(tf.keras.callbacks.Callback):
    """Simple Callback that prints out loss metrics every 10 epochs and
    measures the amount of time per iteration and total training time"""

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            # print(
            #     "Epoch: {} \t Loss: {:.9f} \t Val Loss {:.9f} \t Time: {:.3f}".format(
            #         epoch, logs["loss"], logs["val_loss"], time.time() - self.start_time
            #     )
            # )
            print(
                "Epoch: {} \t Loss: {:.9f} \t Val Loss: {:.9f} \t Time: {:.3f} \t Avg Error: {:.9f}% \t Max Error: {:.9f}%".format(
                    epoch, logs["loss"], logs["val_loss"], time.time() - self.start_time, logs['val_percent_mean'], logs['val_percent_max'])
                )
            
            self.start_time = time.time()
        # if epoch % 1000 == 0:
        #     print("Epoch: {} \t adaptive: {}".format(epoch, logs["adaptive_constant"]))

    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        self.time_delta = np.round(self.end_time - self.train_start, 2)


class GradientCallback(tf.keras.callbacks.Callback):
    """Callback that plots out the gradients for each hidden layer after every 1000 epochs"""

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print(
                "Epoch: {} \t Loss: {:.9f} \t Val Loss {:.9f} \t Time: {:.3f}".format(
                    epoch, logs["loss"], logs["val_loss"], time.time() - self.start_time
                )
            )
            self.start_time = time.time()
        if epoch % 1000 == 0:
            print("Epoch: {} \t adaptive: {}".format(epoch, logs["adaptive_constant"]))
            fig, ax = plt.subplots()
            grad_flat = np.array([])

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
                    epoch, logs["loss"], logs["val_loss"], delta
                )
            )
            self.start_time = time.time()
            self.time_10 = delta

    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        self.time_delta = np.round(self.end_time - self.train_start, 2)
