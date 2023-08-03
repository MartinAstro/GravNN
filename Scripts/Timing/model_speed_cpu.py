import os
import time

os.environ["PATH"] += (
    os.pathsep
    + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"
)
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

if tf.test.gpu_device_name():
    print("GPU found")
else:
    print("No GPU found")

import numpy as np
import pandas as pd

from GravNN.CelestialBodies.Asteroids import Eros
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Networks.Model import load_config_and_model


def generate_test_dataset(x, batch_size):
    x = x.astype("float32")
    dataset = tf.data.Dataset.from_tensor_slices((x,))
    dataset = dataset.shuffle(1000, seed=1234)
    dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.data.experimental.copy_to_device("/gpu:0"))
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()
    # Why Cache is Impt: https://stackoverflow.com/questions/48240573/why-is-tensorflows-tf-data-dataset-shuffle-so-slow
    return dataset


def time_model(positions, model, batch=True):
    start = time.time()
    if batch:
        model.compute_acceleration(positions)
    else:
        for i in range(len(positions)):
            model.compute_acceleration(np.array([positions[i, :]]))
    delta = time.time() - start
    try:
        params = (
            model.mesh.vertices.shape[0] * model.mesh.vertices.shape[1]
            + model.mesh.faces.shape[0]
        )
    except:
        params = model.degree * (model.degree + 1)

    return params, delta


def time_network(positions, config, network, batch=True):
    if batch:
        dataset = generate_test_dataset(positions.astype("float32"), 10000)
        network.predict(dataset)
        start = time.time()
        network.predict(dataset)
    else:
        positions = np.array([positions])
        positions_tensor = tf.Variable(positions)
        network.predict(positions_tensor[:, 0, :])
        start = time.time()
        for i in range(len(positions)):
            network.predict(positions_tensor[:, i, :])

    delta = time.time() - start
    params = config["params"][0]
    return params, delta


def time_models_in_df(df_file, column_name, batch=True):
    params_list = []
    time_list = []

    df = pd.read_pickle(df_file).sort_values(by="params", ascending=True)
    for model_id in df["id"].values:
        tf.keras.backend.clear_session()
        config, model = load_config_and_model(df, model_id_file)
        params, delta = time_network(positions, config, model, batch=batch)
        params_list.append(params)
        time_list.append(delta)
    return pd.DataFrame(data=time_list, index=params_list, columns=[column_name])


positions = np.random.uniform(size=(10000, 3)) * 1e4  # Must be in meters
# positions = np.random.uniform(size=(100,3))*1E4


def conference_timing():
    Earth()
    Eros()

    # #! Run on CPU
    # #* Traditional Network (CPU)
    nn_CPU_df = time_models_in_df("N_10000_rand_study.data", "nn_cpu_time")
    pinn_CPU_df = time_models_in_df("N_10000_rand_PINN_study.data", "pinn_cpu_time")

    df = pd.read_pickle("Data/Dataframes/speed_results_v2.data")
    df = pd.concat([df, nn_CPU_df, pinn_CPU_df])
    df.to_pickle("Data/Dataframes/speed_results_v2.data")


def journal_timing():
    batch = False
    nn_CPU_df = time_models_in_df(
        "Data/Dataframes/traditional_nn_df.data",
        "nn_cpu_time",
        batch,
    )
    pinn_CPU_df = time_models_in_df(
        "Data/Dataframes/pinn_df.data",
        "pinn_cpu_time",
        batch,
    )

    df = pd.read_pickle("Data/Dataframes/speed_results_journal.data")
    df = pd.concat([df, nn_CPU_df, pinn_CPU_df])
    df.to_pickle("Data/Dataframes/speed_results_journal.data")


if __name__ == "__main__":
    # conference_timing()
    journal_timing()
