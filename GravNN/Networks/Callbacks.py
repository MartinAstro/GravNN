"""Custom tensorflow callbacks"""
import time
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit

class SimpleCallback(tf.keras.callbacks.Callback):
    """Simple Callback that prints out loss metrics every 10 epochs and
    measures the amount of time per iteration and total training time"""
    def __init__(self, batch_size, print_interval=10):
        super().__init__()
        self.batch_size = batch_size
        self.print_interval = print_interval

        self.N_train_samples = 0
        self.epoch_loss = 0.0
        self.epoch_percent_mean = 0.0
        self.epoch_percent_max = 0.0

        self.N_val_samples = 0
        self.epoch_val_loss = 0.0
        self.epoch_val_percent_mean = 0.0
        self.epoch_val_percent_max = 0.0

    def incremental_average(self,avg, val, N, M):
        return (val*M + avg*N)/(N+M)

    def on_train_batch_end(self, batch, logs=None):
        self.epoch_loss += logs["loss"]
        self.epoch_percent_mean = self.incremental_average(
            self.epoch_percent_mean, 
            logs['percent_mean'], 
            self.N_train_samples,
            self.batch_size)
        self.N_train_samples += self.batch_size
        self.epoch_percent_max = np.max([logs['percent_max'], self.epoch_percent_max])

    def on_test_batch_end(self, batch, logs=None):
        # 'val_' prefix is not yet appended until after epoch
        self.epoch_val_loss += logs["loss"]
        self.epoch_val_percent_mean = self.incremental_average(
            self.epoch_val_percent_mean, 
            logs['percent_mean'], 
            self.N_val_samples,
            self.batch_size)
        self.N_val_samples += self.batch_size
        self.epoch_val_percent_max = np.max([logs['percent_max'], self.epoch_val_percent_max])

    def on_epoch_begin(self,epoch,logs=None):
        self.N_train_samples = 0
        self.epoch_loss = 0.0
        self.epoch_percent_mean = 0.0
        self.epoch_percent_max = 0.0

        self.N_val_samples = 0
        self.epoch_val_loss = 0.0
        self.epoch_val_percent_mean = 0.0
        self.epoch_val_percent_max = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_interval == 0:
            print(
                "Epoch: {} \t Loss: {:.9f} \t Val Loss: {:.9f} \t Time: {:.3f} \t Avg Error: {:.9f}% \t Max Error: {:.9f}%".format(
                    epoch, 
                    self.epoch_loss, 
                    self.epoch_val_loss, 
                    time.time() - self.start_time, 
                    self.epoch_val_percent_mean*100.0, 
                    self.epoch_val_percent_max*100.0)
                )            
            self.start_time = time.time()
        
        # Overwrite batch logs for epoch logs (to be saved in history obj)
        logs['loss'] = self.epoch_loss
        logs['percent_mean'] = self.epoch_percent_mean
        logs['percent_max'] = self.epoch_percent_max

        logs['val_loss'] = self.epoch_val_loss
        logs['val_percent_mean'] = self.epoch_val_percent_mean
        logs['val_percent_max'] = self.epoch_val_percent_max


    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        self.time_delta = np.round(self.end_time - self.train_start, 2)


class RMSComponentsCallback(tf.keras.callbacks.Callback):
    """Loss Components Callback that prints out RMS component metrics every 100 epochs"""
    def __init__(self, batch_size, print_interval=100):
        super().__init__()
        self.batch_size = batch_size
        self.print_interval = print_interval

        self.N_train_samples = 0
        self.loss_components = [0, 0, 0, 0, 0, 0, 0]

        self.N_val_samples = 0
        self.val_loss_components = [0, 0, 0, 0, 0, 0, 0]

    def incremental_average_loss(self, avg, val, N, M):
        old_avg = [N * i for i in avg]
        new_avg = [sum(col)/len(col) for col in zip(*val)]
        full_new_avg = [M * i for i in new_avg]

        final_avg = []
        for i in range(len(full_new_avg)):
            final_avg.append(full_new_avg[i] + old_avg[i])
        final_final_avg = [(i / (N+M)) for i in final_avg]
        return final_final_avg

    def incremental_average(self, avg, val, N, M):
        np.mean(val)
        return ((N*avg + M*val)/(M+N))

    def on_train_batch_end(self, batch, logs=None):
        self.loss_components = self.incremental_average_loss(
            self.loss_components,
            logs['loss_components'],
            self.N_train_samples,
            self.batch_size
        )
        self.N_train_samples += self.batch_size

    def on_test_batch_end(self, batch, logs=None):
        self.val_loss_components = self.incremental_average_loss(
            self.val_loss_components,
            logs['loss_components'],
            self.N_val_samples,
            self.batch_size
        )
        self.N_val_samples += self.batch_size

    def on_epoch_begin(self,epoch,logs=None):
        self.N_train_samples = 0
        self.loss_components = [0, 0, 0, 0, 0, 0, 0]

        self.N_val_samples = 0
        self.val_loss_components = [0, 0, 0, 0, 0, 0, 0]

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.print_interval == 0:
            print("Loss Components: ")
            print(self.loss_components)
            print("Validation Loss Components: ")
            print(self.val_loss_components)        
        
        # Overwrite batch logs for epoch logs (to be saved in history obj)
        logs['loss_components'] = self.loss_components
        logs['val_loss_components'] = self.val_loss_components

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
