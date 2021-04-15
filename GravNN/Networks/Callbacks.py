
import tensorflow as tf
import time 
import numpy as np
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print("Epoch: {} \t Loss: {:.7f} \t Val Loss {:.7f} \t Time: {:.3f}".format(epoch, logs['loss'], logs['val_loss'], time.time() - self.start_time))
            self.start_time = time.time()
    
    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        self.time_delta = np.round(self.end_time - self.train_start , 2)


class TimingCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            delta = time.time() - self.start_time
            print("Epoch: {} \t Loss: {:.7f} \t Val Loss {:.7f} \t Time: {:.3f}".format(epoch, logs['loss'], logs['val_loss'], delta))
            self.start_time = time.time()
            self.time_10 = delta
    
    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        self.time_delta = np.round(self.end_time - self.train_start , 2)

  