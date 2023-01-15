import os
import copy
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from GravNN.Networks import utils
from GravNN.Networks.Constraints import *
from GravNN.Networks.Annealing import *
from GravNN.Networks.Networks import load_network
from GravNN.Networks.Losses import *
from GravNN.Networks.Schedules import get_schedule
from GravNN.Networks.utils import configure_optimizer
from GravNN.Networks.Callbacks import SimpleCallback
from GravNN.Support.transformations_tf import convert_losses_to_sph
import GravNN

np.random.seed(1234)


class PINNGravityModel(tf.keras.Model):
    # Initialize the class
    def __init__(self, config, network=None):
        """Custom Keras model that encapsulates the actual PINN as well as other relevant
        configuration information, and helper functions. This includes all
        training loops, methods to get all (or specific) outputs from the network, and additional
        optimization methods.

        Args:
            config (dict): hyperparameters and configuration variables needed to initialize the network.
            Consult the Config dictionaries within Network.Configs to get an idea of what options are currently
            implemented.
            network (keras.Model): the actual network that will be trained.
        """
        self.variable_cast = config.get("dtype", [tf.float32])[0]
        super(PINNGravityModel, self).__init__(dtype=self.variable_cast)
        self.config = config

        self.mixed_precision = tf.constant(self.config["mixed_precision"][0], dtype=tf.bool)

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
        for loss_key in self.config['loss_fcns'][0]:
            self.loss_fcn_list.append(get_loss_fcn(loss_key))
        self.w_loss = tf.ones(shape=(3,)) # adaptive weights for ALC

    def init_network(self, network):
        self.training = tf.convert_to_tensor(True, dtype=tf.bool)
        self.test_training = tf.convert_to_tensor(False, dtype=tf.bool)
        if network is None:
            self.network = load_network(self.config)
        else:
            self.network = network

        idx = -1
        for i, layer in enumerate(self.network.layers):
            if layer.name == 'planetary_oblateness_layer':
                idx = i
        self.analytic_idx = idx

    def init_analytic_model(self):
        self.analytic_model = tf.keras.Model(
            inputs=self.network.inputs, 
            outputs=self.network.layers[self.analytic_idx].output
            )

    def init_training_steps(self):
        # jacobian ops (needed in LC loss terms) incompatible with XLA
        if ("L" in self.eval.__name__) or \
           ("C" in self.eval.__name__) or \
           (self.config['init_file'][0] is not None) or \
           (not self.config['jit_compile'][0]):
            self.train_step = self.wrap_train_step_njit
            self.test_step = self.wrap_test_step_njit
        else:
            self.train_step = self.wrap_train_step_jit
            self.test_step = self.wrap_test_step_jit

    def init_annealing(self):
        constants = get_PI_adaptive_constants(self.config['PINN_constraint_fcn'][0])
        self.scale_loss = get_PI_annealing(self.config['PINN_constraint_fcn'][0]) # anneal function
        self.calc_adaptive_constant = get_annealing_fcn(self.config["lr_anneal"][0]) # hold, anneal, custom
        self.adaptive_constant = tf.Variable(constants, dtype=self.variable_cast, trainable=False)
        self.beta = tf.Variable(self.config.get('beta', [0.0])[0], dtype=self.variable_cast)

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
        return {
            "potential" : u, "acceleration" : accel,
            "laplacian" : tf.zeros_like(u),
            "curl" : tf.zeros_like(accel)}
    
    def remove_analytic_model(self, x, y_dict, y_hat_dict):
        y_analytic_dict = self.call_analytic_model(x)
        for key in y_dict.keys():
            y_dict[key] -= y_analytic_dict[key]
            y_hat_dict[key] -= y_analytic_dict[key]
        return y_dict, y_hat_dict

    # Training
    def train_step_fcn(self, data):
        """Method to train the PINN. 
        
        Computes the loss components which may contain dU, da, dL, dC or some combination of these variables. 
        
        Args:
            data (tf.Dataset): training data

        Returns:
            dict: dictionary of metrics passed to the callback.
        """
 
        x, y = data

        y_dict = format_training_data(y, self.constraint)

        with tf.GradientTape(persistent=True) as tape:
            # with tf.GradientTape(persistent=True) as w_loss_tape:
            y_hat_dict = self(x, training=self.training) # [N x (3 or 7)]
            y_dict, y_hat_dict = self.remove_analytic_model(x, y_dict, y_hat_dict)

            if self.config.get('loss_sph', [False])[0]:
                convert_losses_to_sph(
                    x, 
                    y_dict['acceleration'], 
                    y_hat_dict['acceleration']
                    )

            losses = MetaLoss(y_hat_dict, y_dict, self.loss_fcn_list)

            # Don't record the gradients associated with
            # computing adaptive learning rates. 
            # with tape.stop_recording():    
            #     self.w_loss = update_w_loss(
            #         self.w_loss,
            #         self._train_counter, 
            #         losses, 
            #         self.network.trainable_variables, 
            #         w_loss_tape)

            # self.w_loss = tf.constant(1.0)
            loss_i = tf.stack([tf.reduce_mean(loss) for loss in losses.values()],0)
            # loss = tf.reduce_sum(self.w_loss*loss_i)
            loss = tf.reduce_sum(loss_i)
            loss = self.optimizer.get_scaled_loss(loss)
            # del w_loss_tape

        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(gradients)
        del tape

        # The PINN loss doesn't depend on the network's final layer bias, so the gradient is None and throws a warning
        # a = df/dx = d/dx stuff * (W_final x + b_final) = d stuff/dx * w
        # loss = a - a_hat
        # d loss/ d weights = no b_final
        self.optimizer.apply_gradients([
            (grad, var) for (grad, var) in zip(gradients, self.network.trainable_variables) if grad is not None
            ])

        return {
            "loss": loss,
            "percent_mean": tf.reduce_mean(losses.get('percent',[0])),
            "percent_max": tf.reduce_max(losses.get('percent',[0])),
        }  

    def test_step_fcn(self, data):
        x, y = data

        y_dict = format_training_data(y, self.constraint)


        y_hat_dict = self(x, training=self.training) # [N x (3 or 7)]
        y_dict, y_hat_dict = self.remove_analytic_model(x, y_dict, y_hat_dict)

        if self.config.get('loss_sph', [False])[0]:
            convert_losses_to_sph(
                x, 
                y_dict['acceleration'], 
                y_hat_dict['acceleration']
                )

        losses = MetaLoss(y_hat_dict, y_dict, self.loss_fcn_list)
        loss = tf.reduce_sum([tf.reduce_mean(loss) for loss in losses.values()])
        return {"loss": loss, 
                "percent_mean": tf.reduce_mean(losses.get('acceleration_percent',[0])),
                "percent_max": tf.reduce_max(losses.get('acceleration_percent',[0])),
                }

    def train(self, data, initialize_optimizer=True):

        if initialize_optimizer:
            optimizer = configure_optimizer(self.config, mixed_precision=False)
        else:
            optimizer = self.optimizer
        self.compile(optimizer=optimizer, loss="mse")
        
        # Train network
        callback = SimpleCallback(self.config['batch_size'][0], print_interval=10)
        schedule = get_schedule(self.config)

        history = self.fit(
            data.train_data,
            epochs=self.config["epochs"][0],
            verbose=0,
            validation_data=data.valid_data,
            callbacks=[callback, schedule],
            use_multiprocessing=True
        )
        history.history["time_delta"] = callback.time_delta

        return history

    # Post-training API calls 
    @tf.function()
    def compute_potential_tf(self, x):
        x_preprocessor = getattr(self, 'x_preprocessor')
        u_postprocessor = getattr(self, 'u_postprocessor')
        x_network_input = x_preprocessor(x) 
        u_network_output = self.network(x_network_input)
        u_output = u_postprocessor(u_network_output)
        return u_output

    def compute_potential(self, x):
        """Method responsible for returning just the PINN potential.
        Use this method if a lightweight TF execution is desired

        Args:
            x (np.array): Input non-normalized position data (cartesian)

        Returns:
            np.array : PINN generated potential
        """
        x = copy.deepcopy(x)
        x_transformer = self.config["x_transformer"][0]
        u_transformer = self.config["u_transformer"][0]
        x = x_transformer.transform(x)
        u_pred = self.network(x)
        try:
            u_pred = u_transformer.inverse_transform(u_pred)
        except:
            u3_vec = np.zeros(x.shape)
            u3_vec[:] = u_pred
            u_pred = u_transformer.inverse_transform(u3_vec)[:,0]
        return u_pred

    def compute_acceleration(self, x, batch_size=131072):
        """Method responsible for returning the acceleration from the
        PINN gravity model. Use this if a lightweight TF execution is
        desired and other outputs are not required.

        Args:
            x (np.array): Input non-normalized position data (cartesian)

        Returns:
            np.array: PINN generated acceleration
        """
        x_transformer = self.config["x_transformer"][0]
        a_transformer = self.config["a_transformer"][0]
        x = x_transformer.transform(x)

        x = tf.constant(x, dtype=self.variable_cast)
        
        # data = utils.chunks(x, 131072//2)

        if self.is_pinn:
            a_pred = self._pinn_acceleration_output(x)
        else:
            a_pred = self._nn_acceleration_output(x)
        a_pred = a_transformer.inverse_transform(a_pred).numpy()

        return a_pred

    def compute_dU_dxdx(self, x, batch_size=131072):
        """Method responsible for returning the acceleration from the
        PINN gravity model. Use this if a lightweight TF execution is
        desired and other outputs are not required.

        Args:
            x (np.array): Input non-normalized position data (cartesian)

        Returns:
            np.array: PINN generated acceleration
        """
        x_transformer = self.config["x_transformer"][0]
        a_transformer = self.config["a_transformer"][0]
        u_transformer = self.config["u_transformer"][0]
        x = x_transformer.transform(x)

        x = tf.constant(x, dtype=self.variable_cast)
        
        # data = utils.chunks(x, 131072//2)

        if self.is_pinn:
            jacobian = self._pinn_acceleration_jacobian(x)
        else:
            jacobian = self._nn_acceleration_jacobian(x)

        l_star = 1/x_transformer.scale_
        t_star = np.sqrt(a_transformer.scale_*l_star)
        jacobian /= t_star**2
        return jacobian


    # JIT wrappers
    @tf.function(jit_compile=True)
    def wrap_train_step_jit(self, data):
        return self.train_step_fcn(data)
    
    @tf.function(jit_compile=False, experimental_relax_shapes=True)
    def wrap_train_step_njit(self, data):
        return self.train_step_fcn(data)
    
    @tf.function(jit_compile=True)
    def wrap_test_step_jit(self, data):
        return self.test_step_fcn(data)

    @tf.function(jit_compile=False, experimental_relax_shapes=True)
    def wrap_test_step_njit(self, data):
        return self.test_step_fcn(data)


    # saving
    def model_size_stats(self):
        """Method which computes the number of trainable variables in the model as well
        as the binary size of the saved network and adds it to the configuration dictionary.
        """
        size_stats = {
            "params": [count_nonzero_params(self.network)],
            "size": [utils.get_gzipped_model_size(self)],
        }
        self.config.update(size_stats)

    def prep_save(self):
        """Method responsible for timestamping the network, adding the training history to the configuration dictionary, and formatting other variables into the configuration dictionary.
        """
        timestamp = pd.Timestamp(time.time(), unit="s").round("ms").ctime()
        time_JD = pd.Timestamp(timestamp).to_julian_date()

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/Dataframes/", exist_ok=True)
        os.makedirs(f"{self.save_dir}/Networks/", exist_ok=True)
        self.config["timetag"] = timestamp
        self.config["id"] = [time_JD]

        # dataframe cannot take fcn objects so settle on the names and convert to fcn on load 
        activation_type = type(self.config["activation"][0])
        activation_string = self.config["activation"][0] if activation_type == str else self.config["activation"][0].__name__
        self.config["activation"] = [activation_string]
        try:
            self.config["optimizer"] = [self.config["optimizer"][0].__module__]
        except:
            pass

        self.config["PINN_constraint_fcn"] = [self.config["PINN_constraint_fcn"][0]]  # Can't have multiple args in each list
        self.model_size_stats()

    def save_custom(self, df_file=None, custom_data_dir=None, history=None, transformers=None):
        """Add remaining training / model variables into the configuration dictionary, then
        save the config variables into its own pickled file, and potentially add it to an existing
        dataframe defined by `df_file`.

        Args:
            df_file (str or pd.Dataframe, optional): path to dataframe to which the config variables should
            be appended or the loaded dataframe itself. Defaults to None.
        """
        # add final entries to config dictionary
        #time.sleep(np.random.randint(0,5)) # Make the process sleep with hopes that it decreases the likelihood that two networks save at the same time TODO: make this a lock instead.

        try:
            self.history = history
            self.config["x_transformer"][0] = transformers["x"]
            self.config["u_transformer"][0] = transformers["u"]
            self.config["a_transformer"][0] = transformers["a"]
            self.config["a_bar_transformer"][0] = transformers["a_bar"]
        except:
            pass

        # Save network and config information
        
        # the default save / load directory is within the GravNN package. 
        self.save_dir = os.path.dirname(GravNN.__file__) + "/../Data"

        # can specify an alternative save / load directory
        if custom_data_dir is not None:
            self.save_dir = custom_data_dir 

        self.prep_save()

        # convert configuration info to dataframe
        config = dict(sorted(self.config.items(), key=lambda kv: kv[0]))
        df = pd.DataFrame().from_dict(config).set_index("timetag")

        # save network, history, and config to unique network directory
        network_id = self.config['id'][0]
        network_dir = f"{self.save_dir}/Networks/{network_id}/"

        self.network.save(network_dir + "network")
        df.to_pickle(network_dir + "config.data")

        with open(network_dir + "history.data", 'wb') as f:
            pickle.dump(self.history.history,f)
        del self.history

        # save config to preexisting dataframe if requested
        if df_file is not None:
            utils.save_df_row(self.config, f"{self.save_dir}/Dataframes/{df_file}")


    # private functions
    @tf.function(jit_compile=True)
    def _nn_acceleration_output(self, x):
        a = self.network(x) 
        return a
    
    @tf.function()
    def _pinn_acceleration_output(self, x):
        a = pinn_A(self.network, x, training=False)
        return a

    @tf.function(experimental_relax_shapes=True)
    def _pinn_acceleration_jacobian(self, x):
        with tf.GradientTape() as g1:
            g1.watch(x)
            with tf.GradientTape() as g2:
                g2.watch(x)
                u = self.network(x)  # shape = (k,) #! evaluate network
            a = -g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative
        jacobian = g1.batch_jacobian(a,x)
        return jacobian
        
    @tf.function(experimental_relax_shapes=True)
    def _nn_acceleration_jacobian(self,x):
        with tf.GradientTape() as g2:
            g2.watch(x)
            a = self.network(x)  # shape = (k,) #! evaluate network
        jacobian = g2.batch_jacobian(a, x)  # shape = (k,n) #! Calculate first derivative
        return jacobian

def backwards_compatibility(config):
    """Convert old configuration variables to their modern
    equivalents such that they can be imported and tested.

    Args:
        config (dict): old configuration dictionary

    Returns:
        dict: new configuration dictionary
    """
    if float(config["id"][0]) < 2459343.9948726853:
        try:
            if np.isnan(config["PINN_flag"][0]): # nan case
                config["PINN_constraint_fcn"] = [pinn_00]
        except:
            pass
    if float(config["id"][0]) < 2459322.587314815:
        if config["PINN_flag"][0] == "none":
            config["PINN_constraint_fcn"] = [pinn_00]
        elif config["PINN_flag"][0] == "gradient":
            config["PINN_constraint_fcn"] = [pinn_A]
        elif config["PINN_flag"][0] == "laplacian":
            config["PINN_constraint_fcn"] = [pinn_APL]
        elif config["PINN_flag"][0] == "conservative":
            config["PINN_constraint_fcn"] = [pinn_APLC]
    
        if "class_weight" not in config:
            config["class_weight"] = [1.0]

        if "dtype" not in config:
            config["dtype"] = [tf.float32]
    if float(config['id'][0]) < 2459640.439074074:
        config['loss_fcn'] = ['rms_summed']
    if float(config["id"][0]) < 2459628.436423611:
        # Before this date, it was assumed that data would be drawn with SH if planet, and 
        # Polyhedral if asteroid. This is no longer true. 
        if "Planets" in config["planet"][0].__module__:
            config["gravity_data_fcn"] = [GravNN.GravityModels.SphericalHarmonics.get_sh_data]
        else:
            config["gravity_data_fcn"] = [GravNN.GravityModels.Polyhedral.get_poly_data]

    if "eros200700.obj" in config["grav_file"][0]:
        from GravNN.CelestialBodies.Asteroids import Eros
        config['grav_file'] = [Eros().obj_200k]

    if "lr_anneal" not in config:
        config["lr_anneal"] = [False]

    if "mixed_precision" not in config:
        config["use_precision"] = [False]

    return config

def load_config_and_model(model_id, df_file, custom_data_dir=None):
    """Primary loading function for the networks and their
    configuration information.

    Args:
        model_id (float): the timestamp of the desired network to load
        df_file (str or pd.Dataframe): the path to (or dataframe itself) containing the network
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

    # Reinitialize the model
    config = backwards_compatibility(config)
    network = tf.keras.models.load_model(
        f"{data_dir}/Networks/{model_id}/network"
    )
    model = PINNGravityModel(config, network)
    optimizer = utils._get_optimizer(config["optimizer"][0])
    model.compile(optimizer=optimizer, loss="mse") 

    return config, model

def count_nonzero_params(model):
    params = 0
    for v in model.trainable_variables:
        params += tf.math.count_nonzero(v)
    return params.numpy()
