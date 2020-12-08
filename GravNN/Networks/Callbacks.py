
import tensorflow as tf
import time 
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 10 == 0:
            print("Epoch: {} \t Loss: {:.5f} \t Time: {:.3f}".format(epoch, logs['loss'], time.time() - self.start_time))
            self.start_time = time.time()

  