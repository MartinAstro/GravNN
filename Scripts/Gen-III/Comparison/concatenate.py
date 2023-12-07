import glob
import os
import pickle

import pandas as pd
from experiment_setup import setup_experiments

import GravNN


def concatenate_dataframes(glob_str, output_df_name):
    experiments = setup_experiments()
    df_list = []
    for i, file in enumerate(glob.glob(glob_str)):
        idx = int(file.split(".pkl")[0].split("_")[-1])
        exp = experiments[idx]
        model_name = exp["model_name"][0]

        with open(file, "rb") as f:
            data_dict = pickle.load(f)
            data_dict.update(exp)
            for key, value in data_dict.items():
                if not isinstance(value, list):
                    data_dict[key] = [value]
            # df_i = pd.DataFrame(data_dict, index=[idx])
            df_i = pd.DataFrame(data_dict, index=[model_name])
        df_list.append(df_i)
    df = pd.concat(df_list, ignore_index=True)
    df.to_pickle(output_df_name)

    return df


def main():
    GravNN_dir = os.path.abspath(os.path.dirname(GravNN.__file__))
    glob_str = GravNN_dir + "/../Data/Comparison/metrics_*.pkl"
    output_df_name = GravNN_dir + "/../Data/Comparison/all_comparisons.data"
    df = concatenate_dataframes(glob_str, output_df_name)
    print(df)
    print(df.columns)
    pass


if __name__ == "__main__":
    main()
