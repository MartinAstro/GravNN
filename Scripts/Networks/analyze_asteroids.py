import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] ='YES'

import multiprocessing as mp
import pandas as pd
from GravNN.CelestialBodies.Asteroids import Bennu, Eros
from script_utils import save_analysis
from GravNN.Analysis.AsteroidAnalyzer import AsteroidAnalyzer

def main():
    """
    Multiprocess the analysis pipeline for each network within the dataframe
    """
    df_file = 'Data/Dataframes/traditional_w_constraints_annealing.data'
    df_file = "Data/Dataframes/transformers_wo_constraints.data"

    interior_bound = Eros().physical_radius
    exterior_bound = Eros().physical_radius + 10000.0
    args = []
    for idx in range(0,25):
        args.append((idx, df_file, interior_bound, exterior_bound))

    with mp.Pool(1) as pool:
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
    config, model = load_config_and_model(model_id, df)

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
