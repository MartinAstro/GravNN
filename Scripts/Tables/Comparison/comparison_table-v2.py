import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sigfig

import GravNN


def transform_to_colormap(data, **kwargs):
    data_rank = data.rank(method="max")

    # get the colormap
    invert = kwargs.get("invert", True)
    colormap = kwargs.get("colormap", "RdYlGn")
    colormap = kwargs.get("colormap", "Spectral")
    if invert:
        map = plt.get_cmap(colormap).reversed()
    else:
        map = plt.get_cmap(colormap)

    # compute transform constants
    unique_ranks = np.unique(data_rank.values)
    unique_max = unique_ranks[-2]
    unique_min = unique_ranks[0]

    if len(unique_ranks) <= 2:
        unique_max = data_rank.max()
        unique_min = data_rank.min()

    # transform data to 0 - 1
    slope = 1 / (unique_max - unique_min)
    intercept = -slope * data_rank.min()
    data_rank = data_rank.apply(lambda x: slope * x + intercept)
    data_rank = data_rank.apply(lambda x: 1 if x > 1 else x)

    # apply colormap
    data_rank = data_rank.apply(lambda x: map(x))
    return data_rank


def round_values(data, **kwargs):
    def apply_int(x):
        try:
            if np.isinf(x):
                return str(x)
            else:
                return str(int(x))
        except:
            return str(x)

    def apply_round(x):
        if x == "D" or np.isinf(x):
            return x
        else:
            too_big = np.log10(x) > 5
            too_small = np.log(x) < -3
            not_zero = x != 0
            if (too_big or too_small) and not_zero:
                notation = "scientific"
            else:
                notation = "standard"
            rounded_value = sigfig.round(
                x,
                kwargs.get("sig_figs", 2),
                notation=notation,
            )
            return rounded_value

    if data.name == "score" or kwargs.get("int", False):
        data = data.apply(lambda x: apply_int(x))
    else:
        data = data.apply(lambda x: apply_round(x))
    return data


def format_data(data, **kwargs):
    divergent_value = kwargs.get("divergent_value", np.inf)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.apply(lambda x: np.inf if x > divergent_value or np.isnan(x) else x)
    # data = data.apply(lambda x: "D" if x >= divergent_value else x)
    return data


def wrap_metrics_in_color(metric_df, color_df):
    latex_df = copy.deepcopy(metric_df)
    for i in range(latex_df.shape[0]):
        str_value = str(latex_df.iloc[i])
        if str_value == "nan":
            str_value = ""
        color = color_df.iloc[i]
        color_str = "{}".format(",".join([f"{c:.2f}" for c in color[:3]]))
        latex_df.iloc[i] = r"\cellcolor[rgb]{" + color_str + "} " + str_value
    return latex_df


def color_data(
    df,
    column,
    **kwargs,
):
    raw_df = copy.deepcopy(df[column])
    color_df = transform_to_colormap(raw_df, **kwargs)
    formatted_raw_df = format_data(raw_df, **kwargs)
    formatted_raw_df = round_values(formatted_raw_df, **kwargs)
    colored_metric_df = wrap_metrics_in_color(formatted_raw_df, color_df)
    return colored_metric_df


def save_data(df, table_name):
    column_format = "r" + "l" * len(df.columns)
    latex_str = df.to_latex(
        escape=False,
        column_format=column_format,
        multicolumn_format="c",
        multirow=True,
        multicolumn=True,
        bold_rows=False,
    )

    # remove underscores
    latex_str = latex_str.replace("_", " ")

    # save the latex string to file
    filepath = f"/Users/johnmartin/Documents/Research/Papers/Journal/PINN_III/Snippets/{table_name}.tex"
    with open(filepath, "w") as file:
        file.write(latex_str)


def format_model_name(df):
    def append_size(row):
        small_in_name = "Small" in row["model_name"]
        large_in_name = "Large" in row["model_name"]

        small_model = row["num_params"] < 1e3
        point_mass_model = "PM" in row["model_name"]
        if small_in_name or large_in_name:
            row["model_name"] = (
                row["model_name"].replace("_Small", "").replace("_Large", "")
            )
            row["model_name"] = (
                row["model_name"].replace("Small", "").replace("Large", "")
            )

        if point_mass_model:
            return row["model_name"] + " -"

        if small_model:
            return row["model_name"] + " S"  # " Small"
        else:
            return row["model_name"] + " L"  # " Large"

    def abbreviate_poly(row):
        if "POLYHEDRAL" in row["model_name"]:
            row["model_name"] = row["model_name"].replace("POLYHEDRAL", "POLY")
        return row["model_name"]

    def bold_PINN_III(row):
        return row["model_name"]
        if "PINN_III" in row["model_name"]:
            row["model_name"] = r"\textbf{" + row["model_name"] + "}"
        return row["model_name"]

    df["model_name"] = df.apply(lambda row: append_size(row), axis=1)
    df["model_name"] = df.apply(lambda row: abbreviate_poly(row), axis=1)
    df["model_name"] = df.apply(lambda row: bold_PINN_III(row), axis=1)
    return df


def format_column_names(df):
    # Rename all percent values
    for column in df.columns:
        if "percent" in column:
            metric_name = column.replace("percent_", "").capitalize()
            metric_name = r"\makecell{" + metric_name + r" \\ (\%)}"
            if "Extrapolation" in metric_name:
                metric_name = metric_name.replace("Extrapolation", "Extrap.")

            df = df.rename(columns={column: metric_name})

    new_names = {
        "score": "Score",
        "model_name": "Model",
        "num_params": "Params",
        "N_train": "N",
        "acc_noise": r"\makecell{Error \\ (\%)}",
        "pos_error": r"\makecell{Traj. \\ (km)}",
        "dt": r"\makecell{$\Delta t$ \\ (s)}",
        "train_duration": "Train Time (s)",
    }
    df = df.rename(columns=new_names)

    return df


def compute_rank(df, method):
    # ranking columns
    ranking_columns = [
        "percent_planes",
        "percent_interior",
        "percent_exterior",
        "percent_extrapolation",
        "percent_surface",
        "pos_error",
        # "dt",
    ]
    # replace nans with infs
    rank_df = df.replace(np.nan, np.inf)
    rank = rank_df[ranking_columns].rank(method="max")
    if method == "sum":
        df["score"] = rank.sum(axis=1)
    if method == "mean":
        df["score"] = rank.mean(axis=1)
    else:
        KeyError("Method not implemented")

    df = df.replace(np.inf, np.nan)
    return df


def color_metadata(df, latex_df):
    # format the training data
    red_hue = plt.get_cmap("Reds").reversed()(0.8)
    white_hue = plt.get_cmap("Greys")(0)
    gray_hue = plt.get_cmap("Greys")(0.5)

    poly_models = df[df["model_name"].str.contains("POLY")].index
    for index in poly_models:
        df.at[index, "N_train"] = np.nan
        df.at[index, "acc_noise"] = np.nan

    training_data = df["N_train"]
    color_df = training_data.apply(
        lambda x: red_hue if x == 500 else (gray_hue if np.isnan(x) else white_hue),
    )
    training_data_latex = wrap_metrics_in_color(training_data, color_df)
    training_data_latex = training_data_latex.apply(
        lambda x: x.replace("50000.0", "50000").replace("500.0", "500"),
    )
    latex_df["N_train"] = training_data_latex

    training_data = df["acc_noise"] * 100
    color_df = training_data.apply(
        lambda x: red_hue if x == 10.0 else (gray_hue if np.isnan(x) else white_hue),
    )
    training_data_latex = wrap_metrics_in_color(training_data, color_df)
    training_data_latex = training_data_latex.apply(
        lambda x: x.replace("10.0", "10").replace("0.0", "0"),
    )
    latex_df["acc_noise"] = training_data_latex

    return latex_df


def main(rank, score):
    directory = os.path.dirname(GravNN.__file__)
    df = pd.read_pickle(directory + "/../Data/Comparison/all_comparisons.data")

    df = pd.read_pickle(
        directory + "/../Data/Comparison/comparison_metrics_072324.data",
    )

    # drop index 20 and 21 -- duplicate polyhedral models
    df = df.drop(index=[20, 21])
    df = df.reset_index()

    # unpack all values from list container where possible
    for col in df.columns:
        try:
            df[col] = df[col].apply(lambda x: x[0] if isinstance(x, list) else x)
        except:
            pass

    df["pos_error"] /= 1e3
    df["pos_error"] *= (24 * 3600 / 1000) * 1 / (24 * 3600)
    df = compute_rank(df, score)
    df = df.sort_values(by="score")
    latex_df = copy.deepcopy(df)

    conversions = [
        {"column": "percent_planes"},
        {"column": "percent_interior"},
        {"column": "percent_exterior"},
        {"column": "percent_extrapolation"},
        {"column": "percent_surface"},
        {"column": "pos_error"},
        {"column": "dt"},
        {"column": "train_duration"},
        {"column": "score"},
    ]

    # color columns
    for conversion in conversions:
        column = conversion["column"]
        if rank:
            df[column] = df[column].rank(method="max")

        latex_df[column] = color_data(df, **conversion)

    latex_df = color_metadata(df, latex_df)

    column_order = [
        "model_name",
        "num_params",
        "N_train",
        "acc_noise",
        "score",
        "percent_planes",
        "percent_interior",
        "percent_exterior",
        "percent_extrapolation",
        "percent_surface",
        "pos_error",
        # "dt",
        "train_duration",
    ]

    latex_df = latex_df[column_order]

    # assign model_name as index
    latex_df = format_model_name(latex_df)
    latex_df = format_column_names(latex_df)

    drop_columns = ["Params", "Train Time (s)"]
    latex_df = latex_df.drop(columns=drop_columns)
    latex_df = latex_df.set_index("Model")
    if rank:
        latex_df = latex_df.rename(
            columns={
                "Score": r"\makecell{Score \\ (Rank)}",
                r"\makecell{Planes \\ (\%)}": r"\makecell{Planes \\ (Rank)}",
                r"\makecell{Interior \\ (\%)}": r"\makecell{Interior \\ (Rank)}",
                r"\makecell{Exterior \\ (\%)}": r"\makecell{Exterior \\ (Rank)}",
                r"\makecell{Extrap. \\ (\%)}": r"\makecell{Extrap. \\ (Rank)}",
                r"\makecell{Surface \\ (\%)}": r"\makecell{Surface \\ (Rank)}",
                r"\makecell{Traj. \\ (km)}": r"\makecell{Traj. \\ (Rank)}",
            },
        )
    save_data(latex_df, f"table_{rank}_{score}")
    if rank:
        latex_df = latex_df.rename(
            columns={
                r"\makecell{Score \\ (Rank)}": "Score",
                r"\makecell{Planes \\ (Rank)}": r"\makecell{Planes \\ (\%)}",
                r"\makecell{Interior \\ (Rank)}": r"\makecell{Interior \\ (\%)}",
                r"\makecell{Exterior \\ (Rank)}": r"\makecell{Exterior \\ (\%)}",
                r"\makecell{Extrap. \\ (Rank)}": r"\makecell{Extrap. \\ (\%)}",
                r"\makecell{Surface \\ (Rank)}": r"\makecell{Surface \\ (\%)}",
                r"\makecell{Traj. \\ (Rank)}": r"\makecell{Traj. \\ (km)}",
            },
        )

    # Extract the value out of the \cellcolor{} command within the rank column
    # use regex
    regex = r"\\cellcolor\[rgb\]{([\d.,]+)}\s*([\d.]+)"
    latex_df["Score"] = (
        latex_df["Score"].str.extract(regex)[1].astype(float).astype(int)
    )

    # rename a column
    latex_df = latex_df.rename(columns={r"\makecell{Error \\ (\%)}": "Error (\%)"})

    df_subset = latex_df.pivot_table(
        index=latex_df.index,
        columns=["N", "Error (\%)"],
        values="Score",
    )
    # collapse items from the first column into the 5th column
    poly_subset = df_subset[df_subset.columns[0]]
    non_nan_index = poly_subset[poly_subset.notna()].index
    desired_column = df_subset.columns[-1]
    # load the polyhedral model rank into the desired column
    for index in non_nan_index:
        df_subset.at[index, desired_column] = poly_subset.loc[index]

    # drop the polyhedral column
    df_subset = df_subset.drop(columns=df_subset.columns[0])

    # rank each column
    df_subset = df_subset.rank(axis=-0, method="max")
    rank_subset = df_subset.mean(axis=1)
    df_subset["rank"] = rank_subset.astype(int)
    df_subset = df_subset.sort_values(by="rank")
    df_subset = df_subset.drop(columns="rank")

    latex_df_subset = copy.deepcopy(df_subset)

    for column in df_subset.columns:
        latex_df_subset[column] = color_data(df_subset, column, int=True)
    save_data(latex_df_subset, f"table_subset_{rank}_{score}")


if __name__ == "__main__":
    main(rank=False, score="sum")
    main(rank=True, score="sum")
