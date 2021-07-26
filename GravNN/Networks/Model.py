import os
import copy
import time
import numpy as np
from numpy.random.mtrand import laplace
import pandas as pd
import tensorflow as tf

from GravNN.Networks import utils
from GravNN.Networks.Constraints import no_pinn, pinn_A, pinn_AL, pinn_ALC, pinn_A_sph
from GravNN.Networks.Annealing import update_constant
from GravNN.Support.transformations import cart2sph, invert_projection, project_acceleration
np.random.seed(1234)

class CustomModel(tf.keras.Model):
    # Initialize the class
    def __init__(self, config, network):
        super(CustomModel, self).__init__()
        self.config = config
        self.network = network
        self.mixed_precision = tf.constant(self.config['mixed_precision'][0], dtype=tf.bool)
        self.variable_cast = config['dtype'][0]
        self.class_weight = tf.constant(config['class_weight'][0], dtype=tf.float32)

        self.calc_adaptive_constant = utils._get_annealing_fcn(config['lr_anneal'][0])
        tensors = utils._get_acceleration_nondim_constants(config['PINN_constraint_fcn'][0], self.config)
        self.scale_tensor = tensors[0]
        self.translate_tensor = tensors[1]
        PINN_variables = utils._get_PI_constraint(config['PINN_constraint_fcn'][0])
        self.eval = PINN_variables[0]
        self.scale_loss = PINN_variables[1]
        self.adaptive_constant = tf.Variable(PINN_variables[2], dtype=tf.float32)
        self.beta = tf.Variable(self.config['beta'][0], dtype=tf.float32)

    def call(self, x, training=None):
        return self.eval(self.network, x, training)
    
    def compute_loss_components(self, y_hat, y):
        # The accelerations are scaled in proportion to the potential
        # such that the acceleration contribution to the loss is extraordinarily small
        # and doesn't effectively contribute to the loss function.
        # This scaling allows for the loss contribution of the acceleration to be on par with 
        # the loss contribution of the potential (when both are being used)
        y_hat_scaled = (y_hat*self.scale_tensor) + self.translate_tensor# + self.translate_tensor)/self.scale_tensor
        y_scaled = (y*self.scale_tensor) + self.translate_tensor
        loss_components = tf.reduce_mean(tf.square(tf.subtract(y_hat_scaled,y_scaled)), 0)
        # tf.print(loss_components)
        return loss_components

    @tf.function()# jit_compile=True)
    def train_step(self, data):
        x, y = data
        with tf.GradientTape(persistent=True) as tape:
            y_hat = self(x, training=True)
            loss_components = self.compute_loss_components(y_hat, y)
            updated_loss_components = self.scale_loss(loss_components, self.adaptive_constant)
            loss = tf.reduce_sum(tf.stack(updated_loss_components))
            loss = self.optimizer.get_scaled_loss(loss)

        # calculate new adaptive constant
        adaptive_constant = self.calc_adaptive_constant(tape, updated_loss_components, \
                                                                self.adaptive_constant, self.beta, \
                                                                self.trainable_weights)                                                                
        
        #tf.print(adaptive_constant)

        # grad_comp_list = []

        # for loss_comp in updated_loss_components:
        #     gradient_components = tape.gradient(loss_comp, self.network.trainable_variables)
        #     grad_comp_list.append(gradient_components)
        
        gradients = tape.gradient(loss, self.network.trainable_variables)
        gradients = self.optimizer.get_unscaled_gradients(gradients)
        del tape

        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))

        return {'loss' : loss, 'adaptive_constant' : adaptive_constant}#, 'grads' : grad_comp_list}

    @tf.function()#jit_compile=True)
    def test_step(self, data):
        x, y = data
        y_hat = self(x, training=True)
        loss_components = self.compute_loss_components(y_hat, y)
        updated_loss_components = self.scale_loss(loss_components, self.adaptive_constant)
        loss =  tf.reduce_sum(updated_loss_components)
        return {'loss' : loss}
    

    
    def output(self, dataset):
        x, y = dataset
        x = tf.Variable(x, dtype=tf.float32)
        assert self.config['PINN_constraint_fcn'][0] != no_pinn
        assert self.config['PINN_constraint_fcn'][0] != pinn_A_sph # Not implimented yet
        with tf.GradientTape(persistent=True) as g1:
            g1.watch(x)
            with tf.GradientTape() as g2:
                g2.watch(x)
                u = self.network(x) # shape = (k,) #! evaluate network                
            u_x = g2.gradient(u, x) # shape = (k,n) #! Calculate first derivative
        u_xx = g1.batch_jacobian(u_x, x)
        
        laplacian = tf.reduce_sum(tf.linalg.diag_part(u_xx),1, keepdims=True)

        curl_x = tf.math.subtract(u_xx[:,2,1], u_xx[:,1,2])
        curl_y = tf.math.subtract(u_xx[:,0,2], u_xx[:,2,0])
        curl_z = tf.math.subtract(u_xx[:,1,0], u_xx[:,0,1])

        curl = tf.stack([curl_x, curl_y, curl_z], axis=1)
        return u, tf.multiply(-1.0,u_x), laplacian, curl


    def generate_nn_data(self, x):
        x = copy.deepcopy(x)
        x_transformer = self.config['x_transformer'][0]
        a_transformer = self.config['a_transformer'][0]
        u_transformer = self.config['u_transformer'][0]
        if self.config['basis'][0] == 'spherical':
            x = cart2sph(x)
            x[:,1:3] = np.deg2rad(x[:,1:3])

        x = x_transformer.transform(x)
        u_pred, a_pred, laplace_pred, curl_pred = self.output((x,x))
        x_pred = x_transformer.inverse_transform(x)
        u_pred = u_transformer.inverse_transform(u_pred)
        a_pred = a_transformer.inverse_transform(a_pred)

        # TODO: (07/02/21): It's likely that laplace and curl should also be inverse transformed as well
        if self.config['basis'][0] == 'spherical':
            x[:,1:3] = np.rad2deg(x[:,1:3])
            #x = sphere2cart(x)
            a_pred = invert_projection(x, a_pred)
            try:
                a_pred = invert_projection(x, a_pred.astype(float))# numba requires that the types are the same
            except:
                a_pred = invert_projection(x, a_pred.numpy().astype(float))# if a_pred is a tensor
        return {'x' : x_pred,
                'u' : u_pred, 
                'a' : a_pred,
                'laplace' : laplace_pred,
                'curl' : curl_pred}

    def generate_potential(self, x):
        x = copy.deepcopy(x)
        x_transformer = self.config['x_transformer'][0]
        u_transformer = self.config['u_transformer'][0]
        if self.config['basis'][0] == 'spherical':
            x = cart2sph(x)
            x[:,1:3] = np.deg2rad(x[:,1:3])

        x = x_transformer.transform(x)
        u_pred = self.network(x)
        u_pred = u_transformer.inverse_transform(u_pred)
        return u_pred

    # https://pychao.com/2019/11/02/optimize-tensorflow-keras-models-with-l-bfgs-from-tensorflow-probability/
    def optimize(self, dataset):
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
            idx = [] # stitch indices
            part = [] # partition indices

            for i, shape in enumerate(shapes):
                n = np.product(shape)
                idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
                part.extend([i]*n)
                count += n

            part = tf.constant(part)

            @tf.function#(jit_compile=True)
            def assign_new_model_parameters(params_1d):
                """A function updating the model's parameters with a 1D tf.Tensor.

                Args:
                    params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
                """

                params = tf.dynamic_partition(params_1d, part, n_tensors)
                for i, (shape, param) in enumerate(zip(shapes, params)):
                    model.trainable_variables[i].assign(tf.reshape(param, shape))

            # now create a function that will be returned by this factory
            @tf.function#(jit_compile=True)
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
                    U_dummy = tf.zeros_like(train_x[:,0:1])

                    #U_dummy = tf.zeros((tf.divide(tf.size(train_x),tf.constant(3)),1))
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
            value_and_gradients_function=func, initial_position=init_params, max_iterations=2000, tolerance=1e-12)#, parallel_iterations=4)

        func.assign_new_model_parameters(results.position)
        self.history.history = func.history

    def model_size_stats(self):
        size_stats = {
            'params' : [count_nonzero_params(self.network)],
            'size' : [utils.get_gzipped_model_size(self)],
        }
        self.config.update(size_stats)

    def prep_save(self):
        timestamp = pd.Timestamp(time.time(), unit='s').round('s').ctime()  
        self.directory = os.path.abspath('.') +"/Data/Networks/"+ str(pd.Timestamp(timestamp).to_julian_date()) + "/"
        os.makedirs(self.directory, exist_ok=True)
        self.config['timetag'] = timestamp
        self.config['history'] = [self.history.history]
        self.config['id'] = [pd.Timestamp(timestamp).to_julian_date()]
        try:
            self.config['activation'] = [self.config['activation'][0].__name__]
        except:
            pass
        try:
            self.config['optimizer'] = [self.config['optimizer'][0].__module__]
        except:
            pass
        self.model_size_stats()

    def save(self, df_file=None):
        # add final entries to config dictionary
        self.prep_save()

        # convert to dataframe
        config = dict(sorted(self.config.items(), key = lambda kv: kv[0]))
        config['PINN_constraint_fcn'] = [config['PINN_constraint_fcn'][0]]# Can't have multiple args in each list

        df = pd.DataFrame().from_dict(config).set_index('timetag')

        # save network and config to directory
        self.network.save(self.directory + "network")
        df.to_pickle(self.directory + "config.data")

        # save config to preexisting dataframe if requested
        if df_file is not None:
            utils.save_df_row(self.config, df_file)


def backwards_compatibility(config):
    if float(config['id'][0]) < 2459322.587314815:
        if config['PINN_flag'][0] == 'none':
            config['PINN_constraint_fcn'] = [no_pinn]
        elif config['PINN_flag'][0] == 'gradient':
            config['PINN_constraint_fcn'] = [pinn_A]
        elif config['PINN_flag'][0] == 'laplacian':
            config['PINN_constraint_fcn'] = [pinn_APL]
        elif config['PINN_flag'][0] == 'convervative':
            config['PINN_constraint_fcn'] = [pinn_APLC]
        
        if 'class_weight' not in config:
            config['class_weight'] = [1.0]
        
        if 'dtype' not in config:
            config['dtype'] = [tf.float32]
        
    if 'lr_anneal' not in config:
        config['lr_anneal'] = [False]

    
    return config
def load_config_and_model(model_id, df_file):
    # Get the parameters and stats for a given run
    # If the dataframe hasn't been loaded
    if type(df_file) == str:
        config = utils.get_df_row(model_id, df_file)
    else:
        # If the dataframe has already been loaded
        config = df_file[model_id == df_file['id']].to_dict()
        for key, value in config.items():
            config[key] = list(value.values())

    # Reinitialize the model
    if 'mixed_precision' not in config:
        config['use_precision'] = [False]
    
    config = backwards_compatibility(config)
    network = tf.keras.models.load_model(os.path.abspath('.') + "/Data/Networks/"+str(model_id)+"/network")
    model = CustomModel(config, network)
    optimizer = utils._get_optimizer(config['optimizer'][0])
    model.compile(optimizer=optimizer, loss='mse') #! Check that this compile is even necessary

    return config, model

def count_nonzero_params(model):
    params = 0
    for v in model.trainable_variables:
        params += tf.math.count_nonzero(v)
    return params.numpy()
