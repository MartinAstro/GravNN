import numpy as np
import pandas as pd
import pickle
import os
import multiprocessing as mp
from GravNN.Support.Grid import Grid
from GravNN.Visualization.VisualizationBase import VisualizationBase
from GravNN.Visualization.MapVisualization import MapVisualization
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.ReducedGridDist import ReducedGridDist
from GravNN.Support.Statistics import mean_std_median, sigma_mask
from sklearn.utils import shuffle

from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.CelestialBodies.Planets import Earth
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics, get_sh_data

from GravNN.Support.StateObject import StateObject
from GravNN.Regression.Regression import Regression



np.random.seed(1234)

def training_validation_split(X, Y, Z, N_train, N_val, random_state=42):

    X, Y, Z = shuffle(X, Y, Z, random_state=random_state)

    X_train = X[:N_train]
    Y_train = Y[:N_train]
    Z_train = Z[:N_train]

    X_val = X[N_train:N_train+N_val]
    Y_val = Y[N_train:N_train+N_val]
    Z_val = Z[N_train:N_train+N_val]

    return X_train, Y_train, Z_train, X_val, Y_val, Z_val

def regress_sh_model(x, a, planet, max_deg, noise, idx):
    file_name = 'regress_' + str(max_deg) + "_" + str(noise) + "_" + str(idx) 
    directory = os.path.join(os.path.abspath('.') , 'GravNN','Files', 'GravityModels','Regressed')
    os.makedirs(directory, exist_ok=True)
    grav_file = os.path.join(directory, file_name + '.csv')
    regressor = Regression(max_deg, planet, x, a)
    coefficients = regressor.perform_regression(remove_deg=True)
    regressor.save(grav_file)
    return file_name

def regress_nn_model(x, a, x_val, a_val, num_units, pinn):
    import time
    import tensorflow as tf
    from GravNN.Networks.Data import generate_dataset
    from GravNN.Networks.Configs.Default_Configs import get_default_earth_config, get_default_earth_pinn_config
    from GravNN.Networks.Networks import load_network
    from GravNN.Networks.Model import CustomModel
    from GravNN.Networks.Callbacks import CustomCallback

    if pinn:
        config = get_default_earth_pinn_config()
    else:
        config = get_default_earth_config()
    
    config['num_units']= [num_units]

    tf.random.set_seed(0)
    def configure_optimizer(config):
        optimizer = config['optimizer'][0]
        optimizer.learning_rate = config['learning_rate'][0]

        if config['mixed_precision'][0]:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
        else:
            optimizer.get_scaled_loss = lambda x: x
            optimizer.get_unscaled_gradients = lambda x: x
        return optimizer
        
    tf.keras.backend.clear_session()

    if 'num_units' in config:
        for i in range(1, len(config['layers'][0])-1):
            config['layers'][0][i] = config['num_units'][0]

    x_transformer = config['x_transformer'][0]
    a_transformer = config['a_transformer'][0]

    x_scaled = x_transformer.fit_transform(x)
    a_scaled = a_transformer.fit_transform(a)

    x_val_scaled = x_transformer.transform(x_val)
    a_val_scaled = a_transformer.transform(a_val)

    dataset = generate_dataset(x_scaled, a_scaled, 8192*2)
    val_dataset = generate_dataset(x_val_scaled, a_val_scaled, 8192*2)

    network = load_network(config)
    model = CustomModel(config, network)
    optimizer = configure_optimizer(config)
    model.compile(optimizer=optimizer, loss="mse", run_eagerly=False)#, run_eagerly=True)#, metrics=["mae"])

    config['min_delta'] = [1E-9]
    scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                                        monitor='val_loss', factor=0.9, patience=500, verbose=0,
                                        mode='auto', min_delta=config['min_delta'][0], cooldown=0, min_lr=0, 
                                        )
    early_stop = tf.keras.callbacks.EarlyStopping(
                                        monitor='val_loss', min_delta=config['min_delta'][0], patience=1000, verbose=1,
                                        mode='auto', baseline=None, restore_best_weights=True
                                    )
    callback = CustomCallback()
    history = model.fit(dataset, 
                        epochs=config['epochs'][0], 
                        verbose=0,
                        validation_data=val_dataset,
                        callbacks=[callback, scheduler, early_stop])
    model.history = history

    # TODO: Save extra parameters like optimizer.learning_rate
    # Save network and config information
    model.config['time_delta'] = [callback.time_delta]
    model.config['x_transformer'][0] = x_transformer
    model.config['a_transformer'][0] = a_transformer
    #model.save("Data/Dataframes/regressed_models.data")  

    timestamp = pd.Timestamp(time.time(), unit='s').round('s').ctime()
    model.directory = os.path.abspath('.') +"/Data/Networks/"+ str(pd.Timestamp(timestamp).to_julian_date()) + "/"
    os.makedirs(model.directory, exist_ok=True)
    model.network.save(model.directory + "network")
    model.config['timetag'] = timestamp
    model.config['history'] = [model.history.history]
    model.config['id'] = [pd.Timestamp(timestamp).to_julian_date()]
    try:
        model.config['activation'] = [model.config['activation'][0].__name__]
    except:
        pass
    try:
        model.config['optimizer'] = [model.config['optimizer'][0].__module__]
    except:
        pass
    model.model_size_stats()

    config = dict(sorted(config.items(), key = lambda kv: kv[0]))
    df = pd.DataFrame().from_dict(config).set_index('timetag')
    df.to_pickle(model.directory + "config.data")

    return {'id' : config['id'], 'pinn' : pinn, 'num_units': num_units, 'config' : config}

def main():
    planet = Earth()
    model_file = planet.sh_hf_file
    model_deg = 70 #95 # 33
    model_interval = 5
    N_train = 9500
    N_val = 500
    num_models = 10
    trajectory = RandomDist(planet, [planet.radius, planet.radius+420000.0], 1000000)
    
    deg_list = np.arange(3, model_deg, model_interval, dtype=int)
    num_units_list = [10, 20, 30, 40]
    noise_list =  [0, 2]
    model_id_list = np.arange(0, num_models, 1, dtype=int)

    sh_df = pd.DataFrame(index=pd.MultiIndex.from_product([noise_list, deg_list, model_id_list], names=['noise', 'degree', 'id']), columns=['model_identifier'])
    nn_df = pd.DataFrame(index=pd.MultiIndex.from_product([noise_list, num_units_list, model_id_list], names=['noise', 'nodes', 'id']), columns=['model_identifier'])
    pinn_df = pd.DataFrame(index=pd.MultiIndex.from_product([noise_list, num_units_list, model_id_list], names=['noise', 'nodes', 'id']), columns=['model_identifier'])

    df_regressed = "Data/Dataframes/regressed_models_v2.data"

    pool = mp.Pool(6)
    # Generate N number of models from random data 
    for idx in range(num_models):

        # Get randomly shuffled data 
        x, a, u = get_sh_data(trajectory, model_file, max_deg=1000, deg_removed=2, random_state=idx)
        x, a, u, x_val, a_val, u_val = training_validation_split(x, a, u, N_train, N_val)

        # Bias the data with some amount of noise
        for noise in noise_list:
            a_biased = a + noise*np.std(a)*np.random.randn(a.shape[0], a.shape[1])

            # Regress to increasingly high order models 
            for deg in deg_list:
                print(deg)
                sh_model_name = regress_sh_model(x, a_biased, planet, deg, noise, idx)
                sh_df.loc[(noise, deg, idx)] = sh_model_name

            # Train on increasingly high capacity NN
            args = []
            for num_units in num_units_list:
                args.append((x, a_biased, x_val, a_val, num_units, False))
                args.append((x, a_biased, x_val, a_val, num_units, True))

            results = pool.starmap_async(regress_nn_model, args)
            nn_identifiers = results.get()

            for nn in nn_identifiers:

                # Save off config into dataframe
                config = nn['config']
                config = dict(sorted(config.items(), key = lambda kv: kv[0]))
                df = pd.DataFrame().from_dict(config).set_index('timetag')
                try: 
                    df_all = pd.read_pickle(df_regressed)
                    df_all = df_all.append(df)
                    df_all.to_pickle(df_regressed)
                except: 
                    df.to_pickle(df_regressed)
                
                # Save of identifiers for plotting
                if nn['pinn']:
                    pinn_df.loc[(noise, nn['num_units'], idx)] = nn['id']
                else:
                    nn_df.loc[(noise, nn['num_units'], idx)] = nn['id']    

    sh_df.to_pickle("Data/Dataframes/regress_sh.data")
    nn_df.to_pickle("Data/Dataframes/regress_nn.data")
    pinn_df.to_pickle("Data/Dataframes/regress_pinn.data")

if __name__ == "__main__":
    main()

