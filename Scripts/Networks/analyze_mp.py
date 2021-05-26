import multiprocessing as mp
import pandas as pd
import numpy as np
from GravNN.Networks import utils
from GravNN.CelestialBodies.Planets import Earth, Moon
from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.Trajectories.DHGridDist import DHGridDist
from GravNN.Trajectories.FibonacciDist import FibonacciDist
from GravNN.Trajectories.RandomDist import RandomDist
from GravNN.Trajectories.SurfaceDist import SurfaceDist
def get_altitude_list(planet):
    if planet.__class__ == Earth().__class__:
        sh_stats_df = pd.read_pickle("Data/Dataframes/sh_stats_earth_altitude_v2.data")
        alt_list = np.linspace(0, 500000, 50, dtype=float) # Every 10 kilometers above surface
        window = np.array([5, 15, 45, 100, 300]) # Close to surface distribution
        alt_list = np.concatenate([alt_list, window, 420000+window, 420000-window])
        altitudes = np.sort(np.unique(alt_list))
    elif planet.__class__ == Moon().__class__:
        sh_stats_df = pd.read_pickle("Data/Dataframes/sh_stats_moon_altitude.data")
        altitudes = np.linspace(0, 50000, 50, dtype=float) # Every 1 kilometers above surface
    elif planet.__class__ == Bennu().__class__:
        exit("Not implemented yet")   
    else:
        exit("Selected planet not implemented yet")

    return sh_stats_df, altitudes 
def main():
    '''
    Multiprocess the analysis pipeline for each network within the dataframe 
    '''
    df_file = 'Data/Dataframes/hyperparameter_moon_pinn_80_v10.data'#traditional_nn_df.data'
    df_file = 'Data/Dataframes/moon_traditional_nn_df.data'
    df_file = 'Data/Dataframes/moon_pinn_df.data'

    planet = Moon()
    #planet = Earth()

    analyze_altitude = False

    points = 250000 # 64800
    test_trajectories = {
        "Brillouin" : FibonacciDist(planet, planet.radius, points),
        }

    # * Asteroid config
    # planet = Eros()
    # altitudes = np.arange(0, 10000, 1000, dtype=float) 
    # test_trajectories = {
    #     "Brillouin" : DHGridDist(planet, planet.radius, degree=density_deg),
    #     "Surface" : SurfaceDist(planet, planet.model_25k),
    #     "LBO" : DHGridDist(planet, planet.radius+10000.0, degree=density_deg),
    #     }

    i = 0
    args = []
    for i in range(4):
        args.append((i, df_file, planet, analyze_altitude, test_trajectories, points))
        i += 1


    with mp.Pool(6) as pool:
        output = pool.starmap_async(run, args)
        results = output.get()

    df = pd.read_pickle(df_file)
    for result in results:
        model_id = result[0]
        rse_entries = result[1]
        if model_id is None:
            continue
        df = utils.update_df_row(model_id, df, rse_entries, save=False)
    df.to_pickle(df_file)



def run(i, df_file, planet, analyze_altitude, test_trajectories, points):
    import os
    os.environ["PATH"] += os.pathsep + "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v10.1\\extras\\CUPTI\\lib64"
    os.environ["TF_GPU_THREAD_MODE"] ='gpu_private'
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    import pickle
    import sys
    import tensorflow as tf

    from GravNN.Networks.Analysis import Analysis
    from GravNN.Networks.Model import CustomModel, load_config_and_model

    np.random.seed(1234)
    tf.random.set_seed(0)

    if sys.platform == 'win32':
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

    df = pd.read_pickle(df_file)# ! [5:] -- WARN: if you index, then you'll write over the entire dataframe!

    if i >= len(df):
        return (None, None)
    model_id = df['id'].values[i]

    tf.keras.backend.clear_session()
    config, model = load_config_and_model(model_id, df)

    # Analyze the model
    analyzer = Analysis(model, config)
    rse_entries = analyzer.compute_rse_stats(test_trajectories)
    print(rse_entries)

    if analyze_altitude:
        sh_stats_df, altitudes = get_altitude_list(planet)
        alt_df = analyzer.compute_alt_stats(planet, altitudes, points, sh_stats_df)
        alt_df_file = os.path.abspath('.') +"/Data/Networks/"+str(model_id)+"/rse_alt.data"
        alt_df.to_pickle(alt_df_file)

    return (model_id, rse_entries)

if __name__ == '__main__':
    main()
