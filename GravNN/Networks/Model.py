import os
import copy
import time
import numpy as np
from numpy.random.mtrand import laplace
import pandas as pd
import tensorflow as tf

from GravNN.Networks import utils
from GravNN.Networks.Constraints import *
from GravNN.Networks.Annealing import update_constant
import GravNN

np.random.seed(1234)


class CustomModel(tf.keras.Model):
    # Initialize the class
    def __init__(self, config, network):
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
        super(CustomModel, self).__init__(dtype=self.variable_cast)
        self.config = config
        self.network = network
        self.mixed_precision = tf.constant(
            self.config["mixed_precision"][0], dtype=tf.bool
        )
        
        self.calc_adaptive_constant = utils._get_annealing_fcn(config["lr_anneal"][0])
        self.loss_fcn = utils._get_loss_fcn(config['loss_fcn'][0])
        PINN_variables = utils._get_PI_constraint(config["PINN_constraint_fcn"][0])
        self.eval = PINN_variables[0]
        self.scale_loss = PINN_variables[1]
        self.adaptive_constant = tf.Variable(PINN_variables[2], dtype=self.variable_cast)
        self.beta = tf.Variable(self.config.get('beta', [0.0])[0], dtype=self.variable_cast)

        self.is_pinn = tf.cast(self.config["PINN_constraint_fcn"][0] != no_pinn, tf.bool)
        self.is_modified_potential = tf.cast(self.config["PINN_constraint_fcn"][0] == pinn_A_Ur, tf.bool)

        # jacobian ops incompatible with XLA
        if ("L" in self.eval.__name__) or ( "C" in self.eval.__name__) or (config['init_file'][0] is not None):
            self.train_step = self.train_step_no_jit
            self.test_step = self.test_step_no_jit
        else:
            self.train_step = self.train_step_jit
            self.test_step = self.test_step_jit

    def call(self, x, training=None):
        return self.eval(self.network, x, training)

    def cart2sph(self,x, acc_N):
        X = x[:,0]
        Y = x[:,1]
        Z = x[:,2]
        r = tf.norm(x, axis=1)
        theta = tf.atan2(Y,X)
        phi = tf.atan2(tf.sqrt(tf.square(X) + tf.square(Y)),Z)
        spheres = tf.stack((r,theta,phi), axis=1)

        s_phi = tf.sin(phi)
        c_phi = tf.cos(phi)
        s_theta = tf.sin(theta)
        c_theta = tf.cos(theta)

        r_hat = tf.stack((s_phi*c_theta, s_phi*s_theta,c_phi),axis=1)
        theta_hat = tf.stack((c_phi*c_theta, c_phi*s_theta, -s_phi),axis=1)
        phi_hat = tf.stack((-s_theta, c_theta, tf.zeros_like(s_theta)),axis=1)

        BN = tf.reshape(tf.stack((r_hat, theta_hat, phi_hat),1),(-1,3,3))
        acc_N_3d = tf.reshape(acc_N, (-1, 3, 1))

        # apply BN rotation to each acceleration
        acc_B_3d = tf.einsum('bij,bjk->bik', BN, acc_N_3d)
        acc_B = tf.reshape(acc_B_3d,(-1,3))

        return acc_B


    def compute_rms_components(self, y_hat, y):
        """Separate the different loss component terms.

        Args:
            y_hat (tf.Tensor): predicted values
            y (tf.Tensor): true values

        Returns:
            tf.Tensor: loss components for each contribution (i.e. dU, da, dL, dC)
        """
        loss_components = tf.square(tf.subtract(y_hat, y))
        return loss_components

    def compute_percent_error(self, y_hat, y):
        loss_components = tf.norm(tf.subtract(y_hat, y), axis=1)/tf.norm(tf.abs(y),axis=1)*100
        return loss_components

    @tf.function(jit_compile=True)
    def train_step_jit(self, data):
        """Method to train the PINN. First computes the loss components which may contain dU, da, dL, dC
        or some combination of these variables. These component losses are then scaled by the adaptive learning rate (if flag is True), 
        summed, scaled again (if using mixed precision), the adaptive learning rate is then updated, and then backpropagation
        occurs.

        Args:
            data (tf.Dataset): training data

        Returns:
            dict: dictionary of metrics passed to the callback.
        """
        # with tf.xla.experimental.jit_scope(
        #     compile_ops=lambda node_def: 'batch_jacobian' not in node_def.op.lower(), separate_compiled_gradients=True):

        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            y_hat = self(x, training=True)

            # y_hat = self.cart2sph(x, y_hat)
            # y = self.cart2sph(x, y)

            rms_components = self.compute_rms_components(y_hat, y)
            percent_components = self.compute_percent_error(y_hat, y)

            updated_rms_components = self.scale_loss(
                tf.reduce_mean(rms_components,0), self.adaptive_constant
            )

            loss = self.loss_fcn(rms_components, percent_components)
            loss += tf.reduce_sum(self.network.losses)

            loss = self.optimizer.get_scaled_loss(loss)

        # calculate new adaptive constant
        adaptive_constant = self.calc_adaptive_constant(
            tape,
            updated_rms_components,
            self.adaptive_constant,
            self.beta,
            self.trainable_weights,
        )

        # # These lines are needed if using the gradient callback.
        # grad_comp_list = []
        # for loss_comp in updated_loss_components:
        #     gradient_components = tape.gradient(loss_comp, self.network.trainable_variables)
        #     grad_comp_list.append(gradient_components)

        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(gradients)
        del tape

        # The PINN loss doesn't depend on the network's final layer bias, so the gradient is None and throws a warning
        # a = df/dx = d/dx stuff * (W_final x + b_final) = d stuff/dx * w
        # loss = a - a_hat
        # d loss/ d weights = no b_final
        self.optimizer.apply_gradients([
                (grad, var) 
                for (grad, var) in zip(gradients, self.network.trainable_variables) 
                if grad is not None
                ])
        # self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        return {
            "loss": loss,
            "percent_mean": tf.reduce_mean(percent_components),
            "percent_max": tf.reduce_max(percent_components)
            #"adaptive_constant": adaptive_constant,
        }  # , 'grads' : grad_comp_list}
    
    @tf.function(jit_compile=False, experimental_relax_shapes=True)
    def train_step_no_jit(self, data):
        """Method to train the PINN. First computes the loss components which may contain dU, da, dL, dC
        or some combination of these variables. These component losses are then scaled by the adaptive learning rate (if flag is True), 
        summed, scaled again (if using mixed precision), the adaptive learning rate is then updated, and then backpropagation
        occurs.

        Args:
            data (tf.Dataset): training data

        Returns:
            dict: dictionary of metrics passed to the callback.
        """
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            y_hat = self(x, training=True)
            rms_components = self.compute_rms_components(y_hat, y)
            percent_components = self.compute_percent_error(y_hat, y)

            updated_loss_components = self.scale_loss(
                tf.reduce_mean(rms_components,0), self.adaptive_constant
            )
            loss = self.loss_fcn(rms_components, percent_components)
            loss += tf.reduce_sum(self.network.losses)

            loss = self.optimizer.get_scaled_loss(loss)

        # calculate new adaptive constant
        adaptive_constant = self.calc_adaptive_constant(
            tape,
            updated_loss_components,
            self.adaptive_constant,
            self.beta,
            self.trainable_weights,
        )

        # # These lines are needed if using the gradient callback.
        # grad_comp_list = []
        # for loss_comp in updated_loss_components:
        #     gradient_components = tape.gradient(loss_comp, self.network.trainable_variables)
        #     grad_comp_list.append(gradient_components)

        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(gradients)
        del tape

        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        return {
            "loss": loss,
            "percent_mean": tf.reduce_mean(percent_components),
            "percent_max": tf.reduce_max(percent_components)
            #"adaptive_constant": adaptive_constant,
        }  # , 'grads' : grad_comp_list}

    @tf.function(jit_compile=True)
    def test_step_jit(self, data):
        x, y = data
        y_hat = self(x, training=True)

        # y_hat = self.cart2sph(x, y_hat)
        # y = self.cart2sph(x, y)

        rms_components = self.compute_rms_components(y_hat, y)
        percent_components = self.compute_percent_error(y_hat, y)
        updated_rms_components = self.scale_loss(
            tf.reduce_mean(rms_components,0), self.adaptive_constant
        )
        loss = self.loss_fcn(rms_components, percent_components)
        return {"loss": loss, 
                "percent_mean": tf.reduce_mean(percent_components),
                "percent_max": tf.reduce_max(percent_components)
                }

    @tf.function(jit_compile=False, experimental_relax_shapes=True)
    def test_step_no_jit(self, data):
        x, y = data
        y_hat = self(x, training=True)
        rms_components = self.compute_rms_components(y_hat, y)
        percent_components = self.compute_percent_error(y_hat, y)

        updated_loss_components = self.scale_loss(
            tf.reduce_mean(rms_components,0), self.adaptive_constant
        )
        loss = self.loss_fcn(rms_components, percent_components)

        return {"loss": loss, 
                "percent_mean": tf.reduce_mean(percent_components),
                "percent_max": tf.reduce_max(percent_components)
                }

    @tf.function(jit_compile=True)
    def _nn_acceleration_output(self, x):
        a = self.network(x) 
        return a
    
    @tf.function()
    def _pinn_acceleration_output(self, x):
        if self.is_modified_potential:
            a = pinn_A_Ur(self.network, x, training=False)
        else:
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


    def __nn_output(self, dataset):
        x, y = dataset
        x = tf.Variable(x, dtype=self.variable_cast)
        assert self.config["PINN_constraint_fcn"][0] != no_pinn
        with tf.GradientTape(persistent=True) as g1:
            g1.watch(x)
            with tf.GradientTape() as g2:
                g2.watch(x)
                u = self.network(x)  # shape = (k,) #! evaluate network
            u_x = g2.gradient(u, x)  # shape = (k,n) #! Calculate first derivative
        u_xx = g1.batch_jacobian(u_x, x)

        laplacian = tf.reduce_sum(tf.linalg.diag_part(u_xx), 1, keepdims=True)

        curl_x = tf.math.subtract(u_xx[:, 2, 1], u_xx[:, 1, 2])
        curl_y = tf.math.subtract(u_xx[:, 0, 2], u_xx[:, 2, 0])
        curl_z = tf.math.subtract(u_xx[:, 1, 0], u_xx[:, 0, 1])

        curl = tf.stack([curl_x, curl_y, curl_z], axis=1)
        return u, -u_x, laplacian, curl

    def generate_nn_data(
        self,
        x,
    ):
        """Method responsible for generating all possible outputs of the
        PINN gravity model (U, a, L, C). Note that this is an expensive
        calculation due to the second order derivatives.

        TODO: Investigate if this method can be jit complied and be compatible
        with tf.Datasets for increased speed.

        Args:
            x (np.array): Input data (position)

        Returns:
            dict: dictionary containing all input and outputs of the network
        """
        x = copy.deepcopy(x)
        x_transformer = self.config["x_transformer"][0]
        a_transformer = self.config["a_transformer"][0]
        u_transformer = self.config["u_transformer"][0]
        x = x_transformer.transform(x)

        # This is a cumbersome operation as it computes the Hessian for each term
        u_pred, a_pred, laplace_pred, curl_pred = self.__nn_output((x, x))

        x_pred = x_transformer.inverse_transform(x)
        u_pred = u_transformer.inverse_transform(u_pred)
        a_pred = a_transformer.inverse_transform(a_pred)

        # TODO: (07/02/21): It's likely that laplace and curl should also be inverse transformed as well
        return {
            "x": x_pred,
            "u": u_pred,
            "a": a_pred,
            "laplace": laplace_pred,
            "curl": curl_pred,
        }

    def generate_potential(self, x):
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

    #@tf.function(jit_compile=True)
    def generate_acceleration(self, x, batch_size=131072):
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

        # def chunks(lst, n):
        #     """Yield successive n-sized chunks from lst."""
        #     for i in range(0, len(lst), n):
        #         yield tf.data.Dataset.from_tensor_slices(lst[i:i + n])
        
        # batch_size = 131072//2
        # data = chunks(x, batch_size)

        if self.is_pinn:
            a_pred = self._pinn_acceleration_output(x)
        else:
            a_pred = self._nn_acceleration_output(x)
        a_pred = a_transformer.inverse_transform(a_pred)
        return a_pred

    def generate_dU_dxdx(self, x, batch_size=131072):
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

        # def chunks(lst, n):
        #     """Yield successive n-sized chunks from lst."""
        #     for i in range(0, len(lst), n):
        #         yield tf.data.Dataset.from_tensor_slices(lst[i:i + n])
        
        # batch_size = 131072//2
        # data = chunks(x, batch_size)

        if self.is_pinn:
            jacobian = self._pinn_acceleration_jacobian(x)
        else:
            jacobian = self._nn_acceleration_jacobian(x)
        x_scale = x_transformer.scale_
        u_scale = u_transformer.scale_
        scale = x_scale**2/u_scale
        jacobian = jacobian*scale
        return jacobian


    # https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/
    def optimize(self, dataset):
        """L-BFGS optimizer proposed in original PINN paper, but compatable with TF >2.0. Significantly slower
        than adam, and recommended only for fine tuning the networks after initial optimization with adam.

        Args:
            dataset (tf.Dataset): training input and output data

        """
        import tensorflow_probability as tfp

        class History:
            def __init__(self):
                self.history = []

        self.history = History()

        def function_factory(model, loss, train_x, train_y):
            """A factory to create a function required by tfp.optimizer.lbfgs_minimize.

            Args:
                model [in]: an instance of `tf.keras.Model` or its subclasses.
                loss [in]: a function with signature loss_value = loss(pred_y, true_y).
                train_x [in]: the input part of training data.
                train_y [in]: the output part of training data.

            Returns:
                A function that has a signature of:
                    loss_value, gradients = f(model_parameters).
            """

            # obtain the shapes of all trainable parameters in the model
            shapes = tf.shape_n(model.trainable_variables)
            n_tensors = len(shapes)

            # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
            # prepare required information first
            count = 0
            idx = []  # stitch indices
            part = []  # partition indices

            for i, shape in enumerate(shapes):
                n = np.product(shape)
                idx.append(
                    tf.reshape(tf.range(count, count + n, dtype=tf.int32), shape)
                )
                part.extend([i] * n)
                count += n

            part = tf.constant(part)

            @tf.function  # (jit_compile=True)
            def assign_new_model_parameters(params_1d):
                """A function updating the model's parameters with a 1D tf.Tensor.

                Args:
                    params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
                """

                params = tf.dynamic_partition(params_1d, part, n_tensors)
                for i, (shape, param) in enumerate(zip(shapes, params)):
                    model.trainable_variables[i].assign(tf.reshape(param, shape))

            # now create a function that will be returned by this factory
            @tf.function  # (jit_compile=True)
            def f(params_1d):
                """A function that can be used by tfp.optimizer.lbfgs_minimize.

                This function is created by function_factory.

                Args:
                params_1d [in]: a 1D tf.Tensor.

                Returns:
                    A scalar loss and the gradients w.r.t. the `params_1d`.
                """

                # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
                with tf.GradientTape() as tape:
                    # update the parameters in the model
                    assign_new_model_parameters(params_1d)
                    # calculate the loss
                    U_dummy = tf.zeros_like(train_x[:, 0:1])

                    # U_dummy = tf.zeros((tf.divide(tf.size(train_x),tf.constant(3)),1))
                    pred_y = model(train_x, training=True)
                    loss_value = loss(pred_y, train_y)

                # calculate gradients and convert to 1D tf.Tensor
                grads = tape.gradient(loss_value, model.trainable_variables)
                grads = tf.dynamic_stitch(idx, grads)

                # print out iteration & loss
                f.iter.assign_add(1)
                tf.print("Iter:", f.iter, "loss:", loss_value)

                # store loss value so we can retrieve later
                tf.py_function(f.history.append, inp=[loss_value], Tout=[])

                return loss_value, grads

            # store these information as members so we can use them outside the scope
            f.iter = tf.Variable(0)
            f.idx = idx
            f.part = part
            f.shapes = shapes
            f.assign_new_model_parameters = assign_new_model_parameters
            f.history = []

            return f

        inps = np.concatenate([x for x, y in dataset], axis=0)
        outs = np.concatenate([y for x, y in dataset], axis=0)

        # prepare prediction model, loss function, and the function passed to L-BFGS solver

        loss_fun = tf.keras.losses.MeanSquaredError()
        func = function_factory(self, loss_fun, inps, outs)

        # convert initial model parameters to a 1D tf.Tensor
        init_params = tf.dynamic_stitch(func.idx, self.network.trainable_variables)

        # train the model with L-BFGS solver
        results = tfp.optimizer.lbfgs_minimize(
            value_and_gradients_function=func,
            initial_position=init_params,
            max_iterations=2000,
            tolerance=1e-12,
        )  # , parallel_iterations=4)

        func.assign_new_model_parameters(results.position)
        self.history.history = func.history

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
        self.directory = (
            os.path.abspath(".")
            + "/Data/Networks/"
            + str(pd.Timestamp(timestamp).to_julian_date())
            + "/"
        )
        os.makedirs(self.directory, exist_ok=True)
        os.makedirs(os.path.abspath(".") + "/Data/Dataframes/", exist_ok=True)
        self.config["timetag"] = timestamp
        self.config["history"] = [self.history.history]
        self.config["id"] = [pd.Timestamp(timestamp).to_julian_date()]
        try:
            self.config["activation"] = [self.config["activation"][0].__name__]
        except:
            pass
        try:
            self.config["optimizer"] = [self.config["optimizer"][0].__module__]
        except:
            pass
        self.model_size_stats()

    def save(self, df_file=None, checkpoint_callback=None):
        """Add remaining training / model variables into the configuration dictionary, then
        save the config variables into its own pickled file, and potentially add it to an existing
        dataframe defined by `df_file`.

        Args:
            df_file (str or pd.Dataframe, optional): path to dataframe to which the config variables should
            be appended or the loaded dataframe itself. Defaults to None.
        """
        # add final entries to config dictionary
        #time.sleep(np.random.randint(0,5)) # Make the process sleep with hopes that it decreases the likelihood that two networks save at the same time TODO: make this a lock instead.
        self.prep_save()

        # convert to dataframe
        config = dict(sorted(self.config.items(), key=lambda kv: kv[0]))
        config["PINN_constraint_fcn"] = [
            config["PINN_constraint_fcn"][0]
        ]  # Can't have multiple args in each list

        df = pd.DataFrame().from_dict(config).set_index("timetag")

        # save network and config to directory
        if checkpoint_callback == None:
            self.network.save(self.directory + "network")
        df.to_pickle(self.directory + "config.data")

        # save config to preexisting dataframe if requested
        if df_file is not None:
            utils.save_df_row(self.config, df_file)


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
                config["PINN_constraint_fcn"] = [no_pinn]
        except:
            pass
    if float(config["id"][0]) < 2459322.587314815:
        if config["PINN_flag"][0] == "none":
            config["PINN_constraint_fcn"] = [no_pinn]
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

    if "eros200700.obj" in config["grav_file"][0]:
        from GravNN.CelestialBodies.Asteroids import Eros
        config['grav_file'] = [Eros().obj_200k]

    # Before this date, it was assumed that data would be drawn with SH if planet, and 
    # Polyhedral if asteroid. This is no longer true. 
    if float(config["id"][0]) < 2459628.436423611:
        if "Planets" in config["planet"][0].__module__:
            config["gravity_data_fcn"] = [GravNN.GravityModels.SphericalHarmonics.get_sh_data]
        else:
            config["gravity_data_fcn"] = [GravNN.GravityModels.Polyhedral.get_poly_data]

    
    if "lr_anneal" not in config:
        config["lr_anneal"] = [False]
    return config


def load_config_and_model(model_id, df_file):
    """Primary loading function for the networks and their
    configuration information.

    Args:
        model_id (float): the timestamp of the desired network to load
        df_file (str or pd.Dataframe): the path to (or dataframe itself) containing the network
        configuration parameters of interest.

    Returns:
        tuple: configuration/hyperparater dictionary, compiled CustomModel
    """
    # Get the parameters and stats for a given run
    # If the dataframe hasn't been loaded
    if type(df_file) == str:
        config = utils.get_df_row(model_id, df_file)
    else:
        # If the dataframe has already been loaded
        config = df_file[model_id == df_file["id"]].to_dict()
        for key, value in config.items():
            config[key] = list(value.values())

    # Reinitialize the model
    if "mixed_precision" not in config:
        config["use_precision"] = [False]

    config = backwards_compatibility(config)
    network = tf.keras.models.load_model(
        os.path.dirname(GravNN.__file__) + "/../Data/Networks/" + str(model_id) + "/network"
    )
    model = CustomModel(config, network)
    optimizer = utils._get_optimizer(config["optimizer"][0])
    model.compile(
        optimizer=optimizer, loss="mse"
    )  #! Check that this compile is even necessary

    return config, model


def count_nonzero_params(model):
    params = 0
    for v in model.trainable_variables:
        params += tf.math.count_nonzero(v)
    return params.numpy()
