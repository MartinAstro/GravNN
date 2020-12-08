import os
import zipfile
import tempfile
import tensorflow as tf
import tensorflow_model_optimization as tfmot

def get_gzipped_model_size(model):
    # Returns size of gzipped model, in bytes.
    _, keras_file = tempfile.mkstemp('.h5')
    model.save(keras_file, include_optimizer=False)

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(keras_file)

    return os.path.getsize(zipped_file)

def count_nonzero_params(model):
    params = 0
    for v in model.trainable_variables:
        params += tf.math.count_nonzero(v)
    return params
