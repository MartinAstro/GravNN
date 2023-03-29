import multiprocessing as mp
import os

import pandas as pd

from GravNN.Analysis.PlanetAnalyzer import PlanetAnalyzer
from GravNN.CelestialBodies.Planets import Earth
from GravNN.Networks.Model import load_config_and_model
from GravNN.Networks.script_utils import save_analysis
from GravNN.Trajectories import FibonacciDist

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"


def analyze(idx, df, planet, analyze_altitude, test_trajectories, points):
    import tensorflow as tf

    from GravNN.Networks.script_utils import get_altitude_list

    if idx >= len(df):
        return (None, None)

    model_id = df["id"].values[idx]
    tf.keras.backend.clear_session()
    config, model = load_config_and_model(model_id, df)

    analyzer = PlanetAnalyzer(model, config)
    rse_entries = analyzer.compute_rse_stats(test_trajectories, percent=False)

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

    df_file = "Data/Dataframes/earth_revisited_032723.data"
    planet = Earth()

    # df_file = "Data/Dataframes/moon_traditional_nn_df.data"
    # planet = Moon()

    threads = 4
    analyze_altitude = False
    points = 250000  # 64800

    df_name_truth = "Brillouin"  # deg removed 2
    df_name_truth = "Brillouin_deg_n1"  # no deg removed
    test_trajectories = {df_name_truth: FibonacciDist(planet, planet.radius, points)}

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
