import glob
import os

import pandas as pd

import GravNN


def concatenate_dataframes(glob_str, output_df_name):
    df_list = []
    for file in glob.glob(glob_str):
        df_list.append(pd.read_pickle(file))
    df = pd.concat(df_list, ignore_index=True)
    df.to_pickle(output_df_name)
    return df


def main():
    GravNN_dir = os.path.abspath(os.path.dirname(GravNN.__file__))
    suffix_list = [
        "batch_learning",
        "width_depth",
    ]  # , "data_epochs_small", "data_epochs_large", "noise_loss_large", "noise_loss_small"]
    date = "120323"
    for suffix in suffix_list:
        glob_str = GravNN_dir + f"/../Data/Dataframes/ablation_{suffix}*{date}.data"
        output_df_name = (
            GravNN_dir + f"/../Data/Dataframes/all_ablation_{suffix}{date}.data"
        )
        concatenate_dataframes(glob_str, output_df_name)

    pass


if __name__ == "__main__":
    main()
