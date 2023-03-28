import multiprocessing as mp
import os

import pandas as pd

from GravNN.Analysis.PlanetAnalyzer import PlanetAnalyzer
from GravNN.CelestialBodies.Planets import Earth, Moon
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.script_utils import save_analysis
from GravNN.Trajectories import FibonacciDist

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def analyze(idx, df, planet, analyze_altitude, test_trajectories, points):
    from GravNN.Networks.script_utils import get_altitude_list
    from GravNN.Networks.utils import configure_tensorflow

    tf = configure_tensorflow()

    if idx >= len(df):
        return (None, None)

    model_id = df["id"].values[idx]
    tf.keras.backend.clear_session()
    config, model = load_config_and_model(model_id, df)

    analyzer = PlanetAnalyzer(model, config)
    rse_entries = analyzer.compute_rse_stats(test_trajectories, percent=True)

    if analyze_altitude:
        sh_stats_df, altitudes = get_altitude_list(planet)
        analyzer.compute_alt_stats(planet, altitudes, points, sh_stats_df, model_id)

    print(rse_entries)
    return (model_id, rse_entries)


def main():
    """
    Multiprocess the analysis pipeline for each network within the dataframe
    """
    df_file = "Data/Dataframes/earth_trajectory_v2.data"
    planet = Earth()

    df_file = "Data/Dataframes/moon_traditional_nn_df.data"
    planet = Moon()

    threads = 1
    analyze_altitude = False
    points = 250000  # 64800
    test_trajectories = {"Brillouin": FibonacciDist(planet, planet.radius, points)}

    df = pd.read_pickle(df_file)

    args = []
    for idx in range(0, len(df)):
        args.append((idx, df, planet, analyze_altitude, test_trajectories, points))

    with mp.Pool(threads) as pool:
        output = pool.starmap_async(analyze, args)
        results = output.get()

    save_analysis(df_file, results)


if __name__ == "__main__":
    main()
