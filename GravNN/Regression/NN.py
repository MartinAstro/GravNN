import numpy as np
from GravNN.Networks.utils import configure_run_args
from GravNN.Networks.Configs import *
from GravNN.Preprocessors.UniformScaler import UniformScaler



import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

import copy
import numpy as np
from GravNN.Networks.utils import (
    configure_tensorflow,
    set_mixed_precision,
    check_config_combos,
    format_config_combos,
)
from GravNN.Networks.Callbacks import CustomCallback
from GravNN.Networks.Data import get_preprocessed_data, configure_dataset, compute_normalization_layer_constants
from GravNN.Networks.Model import CustomModel
from GravNN.Networks.Networks import load_network
from GravNN.Networks.utils import load_hparams_to_config, configure_optimizer
from GravNN.Networks.Schedules import get_schedule

tf = configure_tensorflow()

np.random.seed(1234)
tf.random.set_seed(0)
# tf.config.run_functions_eagerly(True)
tf.keras.backend.clear_session()
class NN:
    def __init__(self, config):
        # Get data, network, optimizer, and generate model
        compute_normalization_layer_constants(config)

        optimizer = configure_optimizer(config, mixed_precision=None)
        network = load_network(config)
        model = CustomModel(config, network)
        model.compile(optimizer=optimizer, loss="mse")
        self.model = model


    def update(self, rVec, aVec, iterations=5):
        history = self.model.fit(
                        x=rVec,
                        y=aVec,
                        epochs=iterations,
                        verbose=0,
                    )
        self.model.history = history


