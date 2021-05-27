import multiprocessing as mp
import pandas as pd
from GravNN.CelestialBodies.Planets import Earth, Moon
from GravNN.CelestialBodies.Asteroids import Bennu
from GravNN.Trajectories import DHGridDist, FibonacciDist, RandomDist, SurfaceDist
from script_utils import save_analysis


def main():
    """
    Multiprocess the analysis pipeline for each network within the dataframe
    """
    df_file = "Data/Dataframes/useless_05_27_21.data"
    # df_file = 'Data/Dataframes/moon_pinn_df.data'

    #planet = Moon()
    planet = Earth()

    analyze_altitude = False

    points = 250000  # 64800
    test_trajectories = {"Brillouin": FibonacciDist(planet, planet.radius, points)}

    args = []
    for idx in range(4):
        args.append((idx, df_file, planet, analyze_altitude, test_trajectories, points))

    with mp.Pool(1) as pool:
        output = pool.starmap_async(analyze, args)
        results = output.get()

    save_analysis(df_file, results)


def analyze(idx, df_file, planet, analyze_altitude, test_trajectories, points):
    from script_utils import get_altitude_list
    from GravNN.Networks.utils import configure_tensorflow

    tf = configure_tensorflow()
    from GravNN.Networks.Analysis import Analysis
    from GravNN.Networks.Model import load_config_and_model

    df = pd.read_pickle(df_file)
    if idx >= len(df):
        return (None, None)

    model_id = df["id"].values[idx]
    tf.keras.backend.clear_session()
    config, model = load_config_and_model(model_id, df)

    analyzer = Analysis(model, config)
    rse_entries = analyzer.compute_rse_stats(test_trajectories)

    if analyze_altitude:
        sh_stats_df, altitudes = get_altitude_list(planet)
        analyzer.compute_alt_stats(planet, altitudes, points, sh_stats_df, model_id)

    print(rse_entries)
    return (model_id, rse_entries)


if __name__ == "__main__":
    main()
