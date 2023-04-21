import os

import numpy as np
import tensorflow as tf

import GravNN
from GravNN.Networks import utils
from GravNN.Networks.Annealing import *
from GravNN.Networks.Callbacks import SimpleCallback
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

    # Initialization Fcns
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
            if layer.name == "planetary_oblateness_layer":
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
        if "rms" in self.config["loss_fcns"][0]:
            N_weights -= 1 if "a" in constraints else 0

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
        if initialize_optimizer:
            optimizer = configure_optimizer(self.config, mixed_precision=False)
        self.compile(optimizer=optimizer, loss="mse")

        # Train network
        callback = SimpleCallback(self.config["batch_size"][0], print_interval=10)
        schedule = get_schedule(self.config)

        history = self.fit(
            data.train_data,
            epochs=self.config["epochs"][0],
            verbose=0,
            validation_data=data.valid_data,
            callbacks=[callback, schedule],
            use_multiprocessing=True,
        )
        history.history["time_delta"] = callback.time_delta

        return history

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

    @tf.function(
        jit_compile=True,
        input_signature=[tf.TensorSpec(shape=(None, 3), dtype=tf.float32)],
    )
    def compute_potential(self, x):
        x_input = self.x_preprocessor(x)
        u_pred = self.network(x_input, training=False)
        u = self.u_postprocessor(u_pred)
        return u

    @tf.function(jit_compile=True)
    def preprocess(self, x):
        x_input = self.x_preprocessor(x)
        return x_input

    @tf.function(jit_compile=True)
    def postprocess(self, x):
        x_input = self.a_postprocessor(x)
        return x_input

    def _compute_acceleration(self, x):  # , batch_size=131072):
        x_input = self.preprocess(x)
        # if self.is_pinn:
        a_pred = self.eval_batches(self._pinn_acceleration_output, x_input, 131072 // 2)
        # a_pred = self._pinn_acceleration_output(x_input)
        # else:
        #     a_pred = self._nn_acceleration_output(x_input)
        a = self.postprocess(a_pred)
        return a

    @tf.function(jit_compile=False, reduce_retracing=True)
    def compute_acceleration(self, x):
        return self._compute_acceleration(x)  # , batch_size=131072):

    @tf.function(jit_compile=False, reduce_retracing=True)
    def compute_dU_dxdx(self, x, batch_size=131072):
        x = tf.cast(x, dtype=self.variable_cast)
        x_input = self.preprocess(x)
        fcn = self._pinn_acceleration_jacobian
        jacobian = self.eval_batches(fcn, x_input, batch_size)
        x_star = self.x_preprocessor.scale
        a_star = self.a_preprocessor.scale

        l_star = 1 / x_star
        t_star = tf.sqrt(a_star * l_star)
        jacobian /= t_star**2
        return jacobian

    # JIT wrappers
    @tf.function(jit_compile=True)
    def wrap_train_step_jit(self, data):
        return self.train_step_fcn(data)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def wrap_train_step_njit(self, data):
        return self.train_step_fcn(data)

    @tf.function(jit_compile=True)
    def wrap_test_step_jit(self, data):
        return self.test_step_fcn(data)

    @tf.function(jit_compile=False, reduce_retracing=True)
    def wrap_test_step_njit(self, data):
        return self.test_step_fcn(data)

    # private functions
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
            # u = self.network(x, training=False)
        u_x = tf.negative(tape.gradient(u, x_inputs))
        return u_x

    @tf.function(reduce_retracing=True)
    def _pinn_acceleration_jacobian(self, x):
        with tf.GradientTape() as g1:
            g1.watch(x)
            with tf.GradientTape() as g2:
                g2.watch(x)
                u = self.network(x)  # shape = (k,) #! evaluate network
            a = -g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative
        jacobian = g1.batch_jacobian(a, x)
        return jacobian

    @tf.function(reduce_retracing=True)
    def _nn_acceleration_jacobian(self, x):
        with tf.GradientTape() as g2:
            g2.watch(x)
            a = self.network(x)  # shape = (k,) #! evaluate network
        jacobian = g2.batch_jacobian(a, x)
        return jacobian


def load_config_and_model(model_id, df_file, custom_data_dir=None):
    """Primary loading function for the networks and their
    configuration information.

    Args:
        model_id (float): the timestamp of the desired network to load
        df_file (str or pd.Dataframe): the path to (or dataframe itself) containing net
        configuration parameters of interest.

    Returns:
        tuple: configuration/hyperparameter dictionary, compiled PINNGravityModel
    """
    data_dir = f"{os.path.dirname(GravNN.__file__)}/../Data/"
    if custom_data_dir is not None:
        data_dir = custom_data_dir

    # Get the configuration data specified model_id
    if type(df_file) == str:
        # If the config dataframe hasn't been loaded
        df_file_path = f"{data_dir}/Dataframes/{df_file}"
        config = utils.get_df_row(model_id, df_file_path)
    else:
        # If the config dataframe has already been loaded
        config = df_file[model_id == df_file["id"]].to_dict()
        for key, value in config.items():
            config[key] = list(value.values())

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

    # Reinitialize the model
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

    optimizer = utils._get_optimizer(config["optimizer"][0])
    model.compile(optimizer=optimizer, loss="mse")

    return config, model
