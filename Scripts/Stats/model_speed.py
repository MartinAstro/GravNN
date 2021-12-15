import time
import numpy as np
import pandas as pd
import tensorflow as tf
from GravNN.CelestialBodies.Planets import Earth
from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.GravityModels.SphericalHarmonics import SphericalHarmonics
from GravNN.GravityModels.Polyhedral import Polyhedral
from GravNN.Networks.Model import load_config_and_model
import os

os.environ["PATH"] += os.pathsep + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"
os.environ["TF_GPU_THREAD_MODE"] ='gpu_private'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'


def generate_test_dataset(x, batch_size):
    x = x.astype('float32')
    dataset = tf.data.Dataset.from_tensor_slices((x,))
    dataset = dataset.shuffle(1000, seed=1234)
    dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    #Why Cache is Impt: https://stackoverflow.com/questions/48240573/why-is-tensorflows-tf-data-dataset-shuffle-so-slow
    return dataset

def time_model(positions, model, batch=True):
    start = time.time()
    if batch:
        output = model.compute_acceleration(positions)
    else:
        for i in range(len(positions)):
            output = model.compute_acceleration(np.array([positions[i,:]]))
    delta = time.time() - start
    try:
        params = model.mesh.vertices.shape[0]* model.mesh.vertices.shape[1] + model.mesh.faces.shape[0]
    except:
        params = model.degree*(model.degree+1)

    return params, delta

def time_network(positions, config, network, batch=True):
    if batch:
        dataset = generate_test_dataset(positions.astype('float32'), 10000)
        output = network.predict(dataset)
        start = time.time()
        output = network.predict(dataset)
    else:
        positions = np.array([positions])
        positions_tensor = tf.Variable(positions)
        network.predict(positions_tensor[:,0,:])
        start = time.time()
        for i in range(len(positions)):
            output = network.predict(positions_tensor[:,i,:])

    delta = time.time() - start
    params = config['params'][0]
    return params, delta

def time_models_in_df(df_file, column_name, batch=True):
    params_list = []
    time_list = []

    df = pd.read_pickle(df_file).sort_values(by='params', ascending=True)
    for model_id in df['id'].values:
        tf.keras.backend.clear_session()
        config, model = load_config_and_model(model_id, df_file)
        params, delta = time_network(positions, config, model, batch=batch)
        params_list.append(params)
        time_list.append(delta)
    return pd.DataFrame(data=time_list, index=params_list, columns=[column_name])


def time_polyhedral(asteroid, batch):
    poly_params = []
    poly_time = []
    poly_3 = Polyhedral(asteroid, asteroid.model_3k)
    poly_6 = Polyhedral(asteroid, asteroid.model_6k)
    poly_12 = Polyhedral(asteroid, asteroid.model_12k)
    poly_25 = Polyhedral(asteroid, asteroid.model_25k)

    for model in [poly_3, poly_6, poly_12, poly_25]:
        params, delta = time_model(positions, model, batch=batch)
        poly_params.append(params)
        poly_time.append(delta)
    return pd.DataFrame(data=poly_time, index=poly_params, columns=['poly_time'])


def time_spherical_harmonics(planet, batch):
    sh_params = []
    sh_time = []
    for deg in [9, 10, 13, 15, 20, 25, 35, 45, 55, 65, 75, 125, 200, 400]:
        model = SphericalHarmonics(planet.sh_file, deg)
        params, delta = time_model(positions, model, batch=batch)
        if deg == 9:
            continue
        sh_params.append(params)
        sh_time.append(delta)
    sh_df = pd.DataFrame(data=sh_time, index=sh_params, columns=['sh_time'])
    return sh_df

positions = np.random.uniform(size=(10000,3))*1E4# Must be in meters
#positions = np.random.uniform(size=(100,3))*1E4

def conference_timing():
    batch = True
    poly_df = time_polyhedral(Eros(), batch)
    sh_df = time_spherical_harmonics(Earth(), batch)    
    nn_df = time_models_in_df('N_10000_rand_study.data', 'nn_time', batch)
    pinn_df = time_models_in_df('N_10000_rand_PINN_study.data', 'pinn_time', batch)

    df = pd.concat([poly_df, sh_df, nn_df, pinn_df])#, nn_CPU_df, pinn_CPU_df])
    df.to_pickle('Data/speed_results_v2.data')

def journal_timing():
    batch = False

    poly_df = time_polyhedral(Eros(),batch)
    sh_df = time_spherical_harmonics(Earth(),batch)
    nn_df = time_models_in_df('Data/Dataframes/traditional_nn_df.data', 'nn_time', batch)
    pinn_df = time_models_in_df('Data/Dataframes/pinn_df.data', 'pinn_time', batch)

    df = pd.concat([poly_df, sh_df, nn_df, pinn_df])#, nn_CPU_df, pinn_CPU_df])
    print(df)
    df.to_pickle('Data/Dataframes/speed_results_journal.data')

if __name__ == '__main__':
    journal_timing()