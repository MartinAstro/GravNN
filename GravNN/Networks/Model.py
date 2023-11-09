import os

import numpy as np
import pandas as pd
import tensorflow as tf

# tf.config.run_functions_eagerly(True)
import GravNN
from GravNN.Networks import utils
from GravNN.Networks.Annealing import *
from GravNN.Networks.Callbacks import SimpleCallback, get_early_stop
from GravNN.Networks.Constraints import *
from GravNN.Networks.Layers import *
from GravNN.Networks.Losses import *
from GravNN.Networks.Networks import load_network
from GravNN.Networks.Schedules import get_schedule
from GravNN.Networks.utils import configure_optimizer
from GravNN.Support.transformations_tf import convert_losses_to_sph

np.random.seed(1234)


class PINNGravityModel(tf.keras.Model):
    # Initialize the class
    def __init__(self, config, network=None):
        """Custom Keras model that encapsulates the actual PINN as well as other
        relevant configuration information, and helper functions. This includes all
        training loops, methods to get all (or specific) outputs from the network,
        and additional optimization methods.

        Args:
            config (dict): hyperparameters and configuration variables needed to
                            initialize the network. Consult the Config dictionaries
                            within Network.Configs to get an idea of
                            what options are currently implemented.
            network (keras.Model): the actual network that will be trained.
        """
        self.variable_cast = config.get("dtype", [tf.float32])[0]
        super(PINNGravityModel, self).__init__(dtype=self.variable_cast)
        self.config = config

        self.mixed_precision = tf.constant(
            self.config["mixed_precision"][0],
            dtype=tf.bool,
        )

        self.init_network(network)
        self.init_analytic_model()
        self.init_physics_information()
        self.init_loss_fcns()
        self.init_annealing()
        self.init_training_steps()
        self.init_preprocessing_layers()

    # Initialization Fcns
    def init_preprocessing_layers(self):
        x_transformer = self.config["x_transformer"][0]
        u_transformer = self.config["u_transformer"][0]
        a_transformer = self.config["a_transformer"][0]

        self.x_preprocessor = PreprocessingLayer(
            x_transformer.min_,
            x_transformer.scale_,
            self.dtype,
        )  # normalizing layer
        self.u_postprocessor = PostprocessingLayer(
            u_transformer.min_,
            u_transformer.scale_,
            self.dtype,
        )  # unnormalize layer
        self.a_preprocessor = PreprocessingLayer(
            a_transformer.min_,
            a_transformer.scale_,
            self.dtype,
        )  # normalizing layer
        self.a_postprocessor = PostprocessingLayer(
            a_transformer.min_,
            a_transformer.scale_,
            self.dtype,
        )  # unormalizing layer

    def init_physics_information(self):
        self.constraint = self.config["PINN_constraint_fcn"][0]
        self.eval = get_PI_constraint(self.constraint)
        self.is_pinn = tf.cast(self.constraint != pinn_00, tf.bool)

    def init_loss_fcns(self):
        self.loss_fcn_list = []
        for loss_key in self.config["loss_fcns"][0]:
            self.loss_fcn_list.append(get_loss_fcn(loss_key))

    def init_network(self, network):
        self.training = tf.convert_to_tensor(True, dtype=tf.bool)
        self.test_training = tf.convert_to_tensor(False, dtype=tf.bool)
        self.network = network
        if network is None:
            self.network = load_network(self.config)

        # Determine layer idx of analytic model
        self.analytic_idx = -1
        for idx, layer in enumerate(self.network.layers):
            if layer.name == "analytic_model_layer":
                self.analytic_idx = idx

    def init_analytic_model(self):
        self.analytic_model = tf.keras.Model(
            inputs=self.network.inputs,
            outputs=self.network.layers[self.analytic_idx].output,
        )

    def init_training_steps(self):
        # default to XLA compiled training
        self.train_step = self.wrap_train_step_jit
        self.test_step = self.wrap_test_step_jit

        # fall back if jacobian ops are needed (LC loss terms)
        # as they are incompatible with XLA
        if (
            ("L" in self.eval.__name__)
            or ("C" in self.eval.__name__)
            or (not self.config["jit_compile"][0])
        ):
            self.train_step = self.wrap_train_step_njit
            self.test_step = self.wrap_test_step_njit

    def init_annealing(self):
        anneal_loss = self.config["lr_anneal"][0]
        if anneal_loss:  # currently not jit compatible
            self.config["jit_compile"] = [False]
        self.update_w_fcn = get_annealing_fcn(anneal_loss)
        constraints = self.config["PINN_constraint_fcn"][0].split("_")[1]
        N_constraints = len(constraints)
        N_losses = len(self.config["loss_fcns"][0])
        N_weights = N_constraints * N_losses

        # Remove weights for percent error on L or C
        if "percent" in self.config["loss_fcns"][0]:
            N_weights -= 1 if "l" in constraints else 0
            N_weights -= 1 if "c" in constraints else 0

        # Remove weights for RMS part of A constraint
        # if "rms" in self.config["loss_fcns"][0]:
        #     N_weights -= 1 if "a" in constraints else 0

        constants = list(np.ones((N_weights,)))
        self.w_loss = tf.Variable(constants, dtype=self.dtype, trainable=False)

    def set_training_kwarg(self, training):
        self.training = tf.convert_to_tensor(training, dtype=tf.bool)

    # Model call functions
    def call(self, x, training):
        return self.eval(self.network, x, training)

    def call_analytic_model(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            u = self.analytic_model(x)
        accel = -tape.gradient(u, x)
        return OrderedDict(
            {
                "potential": u,
                "acceleration": accel,
            },
        )

    def remove_analytic_model(self, x, y_dict, y_hat_dict):
        if self.config["fuse_models"][0]:
            y_analytic_dict = self.call_analytic_model(x)
            for key in y_dict.keys() & y_analytic_dict.keys():
                y_dict[key] -= y_analytic_dict[key]
                y_hat_dict[key] -= y_analytic_dict[key]
        return y_dict, y_hat_dict

    # Training
    def train_step_fcn(self, data):
        x, y = data

        y_dict = format_training_data(y, self.constraint)

        with tf.GradientTape(persistent=True) as tape:
            # with tf.GradientTape(persistent=True) as w_loss_tape:
            y_hat_dict = self(x, training=self.training)  # [N x (3 or 7)]
            y_dict, y_hat_dict = self.remove_analytic_model(x, y_dict, y_hat_dict)

            if self.config.get("loss_sph", [False])[0]:
                convert_losses_to_sph(
                    x,
                    y_dict["acceleration"],
                    y_hat_dict["acceleration"],
                )

            losses = MetaLoss(y_hat_dict, y_dict, self.loss_fcn_list)
            loss_i = tf.stack([tf.reduce_mean(loss) for loss in losses.values()], 0)
            loss = tf.reduce_sum(self.w_loss * loss_i)
            loss = self.optimizer.get_scaled_loss(loss)
            # tf.print(loss_i)
            # compute a subset of the losses for w_loss
            # update. Needs to be selected within tape
            # for well defined gradients.
            # This must be computed separately b/c just
            # indexing the loss, the jacobian will compute
            # through the entire loss function which causes
            # massive spike in RAM.
            losses_subset = compute_loss_subset(
                y_hat_dict,
                y_dict,
                self.loss_fcn_list,
            )

        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(gradients)

        # update the weights
        self.update_w_fcn(
            self.w_loss,
            self._train_counter,
            losses_subset,
            self.network.trainable_variables,
            tape,
        )
        del tape

        self.optimizer.apply_gradients(
            [
                (grad, var)
                for (grad, var) in zip(gradients, self.network.trainable_variables)
                if grad is not None
            ],
        )

        return {
            "w_loss": loss,
            "loss": tf.reduce_sum(loss_i),
            "percent_mean": tf.reduce_mean(losses.get("acceleration_percent", [0])),
            "percent_max": tf.reduce_max(losses.get("acceleration_percent", [0])),
        }

    def test_step_fcn(self, data):
        x, y = data

        y_dict = format_training_data(y, self.constraint)
        y_hat_dict = self(x, training=self.training)  # [N x (3 or 7)]
        y_dict, y_hat_dict = self.remove_analytic_model(x, y_dict, y_hat_dict)

        if self.config.get("loss_sph", [False])[0]:
            convert_losses_to_sph(
                x,
                y_dict["acceleration"],
                y_hat_dict["acceleration"],
            )

        losses = MetaLoss(y_hat_dict, y_dict, self.loss_fcn_list)
        loss = tf.reduce_sum([tf.reduce_mean(loss) for loss in losses.values()])
        return {
            "loss": loss,
            "percent_mean": tf.reduce_mean(losses.get("acceleration_percent", [0])),
            "percent_max": tf.reduce_max(losses.get("acceleration_percent", [0])),
        }

    def train(self, data, initialize_optimizer=True):
        optimizer = self.optimizer
        if initialize_optimizer and optimizer is None:
            optimizer = configure_optimizer(self.config, mixed_precision=False)
            self.compile(optimizer=optimizer, loss="mse")

        # Train network
        callback = SimpleCallback(
            self.config["batch_size"][0],
            print_interval=self.config.get("print_interval", [10])[0],
        )
        schedule = get_schedule(self.config)

        callbacks = [callback, schedule]
        if self.config.get("early_stop", [False])[0]:
            early_stop = get_early_stop(self.config)
            callbacks.append(early_stop)

        history = self.fit(
            data.train_data,
            epochs=self.config["epochs"][0],
            verbose=0,
            validation_data=data.valid_data,
            callbacks=callbacks,
            use_multiprocessing=True,
        )
        history.history["time_delta"] = callback.time_delta

        return history

    # JIT wrappers
    @tf.function(jit_compile=True, reduce_retracing=True)
    def wrap_train_step_jit(self, data):
        return self.train_step_fcn(data)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def wrap_train_step_njit(self, data):
        return self.train_step_fcn(data)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def wrap_test_step_jit(self, data):
        return self.test_step_fcn(data)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def wrap_test_step_njit(self, data):
        return self.test_step_fcn(data)

    def eval_batches(self, fcn, x, batch_size):
        data = utils.chunks(x, batch_size)
        y = []
        for x_batch in data:
            y_batch = fcn(x_batch)
            if len(y) == 0:
                y = y_batch
            else:
                y = tf.concat((y, y_batch), axis=0)
        return y

    @tf.function(jit_compile=False, reduce_retracing=True)
    def compute_potential(self, x):
        x_input = self.x_preprocessor(x)
        u_pred = self.network(x_input, training=False)
        u = self.u_postprocessor(u_pred)
        return u

    @tf.function(jit_compile=False, reduce_retracing=True)
    def compute_disturbing_potential(self, x):
        x_input = self.x_preprocessor(x)
        u_pred = self.network(x_input, training=False)
        u_analytic = self.analytic_model(x_input)
        u_dist = u_pred - u_analytic
        u = self.u_postprocessor(u_dist)
        return u

    # @tf.function(jit_compile=True)
    def preprocess(self, x):
        x_input = self.x_preprocessor(x)
        return x_input

    @tf.function(jit_compile=True)
    def postprocess(self, x):
        x_input = self.a_postprocessor(x)
        return x_input

    @tf.function(jit_compile=False, reduce_retracing=True)
    def compute_acceleration(self, x):
        return self._compute_acceleration(x)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def compute_dU_dxdx(self, x):
        return self._compute_dU_dxdx(x)

    # private functions
    def _compute_acceleration(self, x):
        x_input = self.preprocess(x)
        # fcn = self._pinn_acceleration_output
        # a_pred = self.eval_batches(fcn, x_input, 131072 // 2)
        a_pred = self._pinn_acceleration_output(x_input)
        a = self.postprocess(a_pred)
        return a

    def _compute_dU_dxdx(self, x):
        x = tf.cast(x, dtype=self.variable_cast)
        x_input = self.preprocess(x)
        fcn = self._pinn_acceleration_jacobian
        jacobian = self.eval_batches(fcn, x_input, 131072 // 2)
        # jacobian = self._pinn_acceleration_jacobian(x_input)
        x_star = tf.cast(self.x_preprocessor.scale, dtype=self.variable_cast)
        a_star = tf.cast(self.a_preprocessor.scale, dtype=self.variable_cast)

        l_star = 1 / x_star
        t_star = tf.sqrt(a_star * l_star)
        jacobian /= t_star**2
        return jacobian

    def _nn_acceleration_output(self, x):
        a = pinn_00(self.network, x, training=False)["acceleration"]
        return a

    @tf.function(jit_compile=True)
    def _network_potential(self, x, training):
        return self.network(x, training=training)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def _pinn_acceleration_output(self, x):
        x_inputs = x
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x_inputs)
            u = self._network_potential(x_inputs, training=False)
        u_x = tf.negative(tape.gradient(u, x_inputs))
        return u_x

    # @tf.function(jit_compile=False, reduce_retracing=True)
    def _pinn_acceleration_jacobian(self, x):
        x_inputs = x
        with tf.GradientTape(watch_accessed_variables=False) as g1:
            g1.watch(x_inputs)
            with tf.GradientTape(watch_accessed_variables=False) as g2:
                g2.watch(x_inputs)
                u = self.network(
                    x_inputs,
                    training=False,
                )  # shape = (k,) #! evaluate network
            # shape = (k,n) #! Calculate first derivative
            a = tf.negative(g2.gradient(u, x_inputs))
        jacobian = g1.batch_jacobian(a, x_inputs)
        return jacobian

    @tf.function(jit_compile=False, reduce_retracing=True)
    def _nn_acceleration_jacobian(self, x):
        with tf.GradientTape() as g2:
            g2.watch(x)
            a = self.network(x)  # shape = (k,) #! evaluate network
        jacobian = g2.batch_jacobian(a, x)
        return jacobian


def load_config_and_model(
    df_file,
    model_id=None,  # timestamp of model
    idx=-1,  # index in dataframe
    custom_dtype=None,
    only_weights=False,
):
    """Primary loading function for the networks and their
    configuration information.

    Args:
        model_id (float): the timestamp of the desired network to load
        df_file (str or pd.Dataframe): the path to (or dataframe itself) containing net
        configuration parameters of interest.

    Returns:
        tuple: configuration/hyperparameter dictionary, compiled PINNGravityModel
    """

    # RESOLVE PATHS
    # assume model is saved in GravNN/Data/ directory
    data_dir = f"{os.path.dirname(GravNN.__file__)}/../Data"

    # LOAD CONFIG
    # Get the configuration data specified model_id
    if type(df_file) == str:
        df_file_basename = os.path.basename(df_file)

        # If there is a /Data/ dir that is within the df_path, use it.
        if "/Data/" in df_file and os.path.isabs(df_file):
            data_dir = df_file.split("/Data/")[0] + "/Data"

        # If the config dataframe hasn't been loaded
        df_file_path = f"{data_dir}/Dataframes/{df_file_basename}"
        print("Loading from: ", df_file_path)
        df = pd.read_pickle(df_file_path)

    elif type(df_file) == pd.DataFrame:
        df = df_file
    else:
        raise Exception("Invalid df_file type")

    # Get the configuration data specified model_id
    if model_id is None:
        config = df.iloc[idx].to_dict()
        model_id = config["id"]
        print(f"INFO: Model ID not specified, loading idx={idx} (model ID={model_id}))")
    else:
        config = df[model_id == df["id"]].to_dict()
        print(f"INFO: Loading Model ID={model_id})")

    for key, value in config.items():
        try:
            config[key] = list(value.values())
        except Exception:
            config[key] = [value]

    # remove nan's
    drop_keys = []
    for key, value in config.items():
        try:
            if np.isnan(value[0]):
                drop_keys.append(key)
        except Exception:
            pass
    for key in drop_keys:
        config.pop(key)

    # Change model dtype if specified
    if custom_dtype is not None:
        config["dtype"] = [custom_dtype]

    # HACK: Fix grav file if necessary
    grav_file = config.get("grav_file", [None])[0]
    obj_file = grav_file if grav_file is not None else config["obj_file"][0]
    sh_file = grav_file if grav_file is not None else config["sh_file"][0]
    config["obj_file"] = [obj_file]
    config["sh_file"] = [sh_file]

    if only_weights:
        model = PINNGravityModel(config)
        weights_save_dir = config.get("save_dir", [data_dir])[0]
        # attempt to find the weights_save_dir with the installed GravNN dir
        if "/projects/joma5012" in weights_save_dir:
            weights_save_dir_parts = weights_save_dir.split(
                "/projects/joma5012/GravNN/",
            )
            if len(weights_save_dir_parts) == 1:
                weights_save_dir = os.path.dirname(GravNN.__file__) + "/../Data"

            else:
                weights_save_dir = (
                    os.path.dirname(GravNN.__file__)
                    + "/../.."
                    + weights_save_dir_parts[-1]
                )

        try:
            model.network.load_weights(
                f"{weights_save_dir}/Networks/{model_id}/weights",
            )
        except:
            new_dir = f"{weights_save_dir}/Networks/{model_id}/weights".replace(
                "GravNN",
                "StatOD",
            )
            model.network.load_weights(new_dir)

    else:
        # Reinitialize the model
        try:
            network = tf.keras.models.load_model(
                f"{data_dir}/Networks/{model_id}/network",
            )
        except:
            data_dir = data_dir.replace("GravNN", "StatOD")
            data_dir = data_dir.replace("ML_Gravity", "StatOD")
            network = tf.keras.models.load_model(
                f"{data_dir}/Networks/{model_id}/network",
            )
        model = PINNGravityModel(config, network)

    x_transformer = config["x_transformer"][0]
    u_transformer = config["u_transformer"][0]
    a_transformer = config["a_transformer"][0]

    model.x_preprocessor = PreprocessingLayer(
        x_transformer.min_,
        x_transformer.scale_,
        model.dtype,
    )  # normalizing layer
    model.u_postprocessor = PostprocessingLayer(
        u_transformer.min_,
        u_transformer.scale_,
        model.dtype,
    )  # unnormalize layer
    model.a_preprocessor = PreprocessingLayer(
        a_transformer.min_,
        a_transformer.scale_,
        model.dtype,
    )  # normalizing layer
    model.a_postprocessor = PostprocessingLayer(
        a_transformer.min_,
        a_transformer.scale_,
        model.dtype,
    )  # unormalizing layer

    optimizer = configure_optimizer(
        config,
        None,
    )
    model.compile(optimizer=optimizer, loss="mse")

    return config, model
