import glob
import os

import pandas as pd

import GravNN


def transfer_df(df_file):
    # Copy the dataframe and its associated networks into a single directory for easy transfer
    gravNN_dir = os.path.dirname(GravNN.__file__)
    os.makedirs(f"{gravNN_dir}/../Data/Transfer/", exist_ok=True)

    df = pd.read_pickle(df_file)

    # get network ids
    ids = df.id.values

    # get network files
    for id in ids:
        # for every network id, get the copy the associated directory in Data/Networks/
        files = glob.glob(f"{gravNN_dir}/../Data/Networks/{id}/*")
        os.makedirs(f"{gravNN_dir}/../Data/Transfer/{id}/", exist_ok=True)

        # copy the directory (keeping the id) to Data/Transfer/
        for file in files:
            os.system(f"cp -r {file} {gravNN_dir}/../Data/Transfer/{id}/")

        # copy the dataframe to Data/Transfer/
        os.system(f"cp {df_file} {gravNN_dir}/../Data/Transfer/")


def main():
    gravNN_dir = os.path.dirname(GravNN.__file__)
    transfer_df(f"{gravNN_dir}/../Data/Dataframes/pinn_primary_figure_II.data")
    transfer_df(f"{gravNN_dir}/../Data/Dataframes/pinn_primary_figure_III.data")


if __name__ == "__main__":
    main()
