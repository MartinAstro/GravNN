import os

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import multiprocessing as mp

import pandas as pd

from GravNN.CelestialBodies.Asteroids import Bennu, Eros
from GravNN.Networks.script_utils import save_analysis


def main():
    """
    Multiprocess the analysis pipeline for each network within the dataframe
    """

    df_file = "Data/Dataframes/transformer_wo_annealing.data"

    df_file = "Data/Dataframes/eros_official_noise_transformer_no_annealing.data"
    df_file = "Data/Dataframes/eros_official_noise_transformer_annealing.data"

    df_file = "Data/Dataframes/eros_official_transformer_pinn_40.data"

    interior_bound = Eros().radius
    exterior_bound = Eros().radius * 3

    # df_file = "Data/Dataframes/bennu_traditional_wo_annealing.data"
    # df_file = "Data/Dataframes/bennu_official_w_noise_2.data"
    df_file = "Data/Dataframes/bennu_residual.data"
    interior_bound = Bennu().radius
    exterior_bound = Bennu().radius * 3

    args = []
    for idx in range(0, 45):
        args.append((idx, df_file, interior_bound, exterior_bound))

    with mp.Pool(4) as pool:
        output = pool.starmap_async(analyze, args)
        results = output.get()

    save_analysis(df_file, results)


def analyze(idx, df_file, interior_bound, exterior_bound):
    from GravNN.Networks.utils import configure_tensorflow

    tf = configure_tensorflow()
    from GravNN.Analysis.AsteroidAnalyzer import AsteroidAnalyzer
    from GravNN.Networks.Model import load_config_and_model

    df = pd.read_pickle(df_file)
    if idx >= len(df):
        return (None, None)

    model_id = df["id"].values[idx]
    tf.keras.backend.clear_session()
    config, model = load_config_and_model(df, model_id)

    analyzer = AsteroidAnalyzer(model, config, interior_bound, exterior_bound)
    surface_stats = analyzer.compute_surface_stats()
    interior_stats = analyzer.compute_interior_stats()
    exterior_stats = analyzer.compute_exterior_stats()

    stats = {}
    stats.update(surface_stats)
    stats.update(interior_stats)
    stats.update(exterior_stats)

    print(stats)
    return (model_id, stats)


if __name__ == "__main__":
    main()
