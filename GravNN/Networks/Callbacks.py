
import tensorflow as tf
import time 
import numpy as np
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print("Epoch: {} \t Loss: {:.5f} \t Val Loss {:.5f} \t Time: {:.3f}".format(epoch, logs['loss'], logs['val_loss'], time.time() - self.start_time))
            self.start_time = time.time()
    
    def on_train_begin(self, logs=None):
        self.train_start = time.time()
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        self.end_time = time.time()
        self.time_delta = np.round(self.end_time - self.train_start , 2)

  