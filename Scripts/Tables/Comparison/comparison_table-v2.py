import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sigfig

import GravNN

GRAY_RGBA = plt.get_cmap("gray")(0.9)
RED_RGBA = plt.get_cmap("RdYlGn").reversed()(1.0)


def transform_to_colormap(data, min_value, mid_value=None, max_value=None):
    # throw error if mid_value and max_value are both provided
    if mid_value is not None and max_value is not None:
        raise ValueError("Cannot provide both mid_value and max_value")

    # generate a linear transform using either the min and mid or min and max value
    if mid_value is None:
        slope = 1 / (max_value - min_value)
        intercept = -slope * min_value
    else:
        slope = 0.5 / (mid_value - min_value)
        intercept = -slope * min_value

    # apply the linear transform to the data
    data = data.apply(lambda x: slope * x + intercept)

    # clip the data after 1 and before 0
    data = data.apply(lambda x: 1 if x > 1 else x)
    data = data.apply(lambda x: 0 if x < 0 else x)

    # apply the colormap to the data
    data = data.apply(lambda x: plt.get_cmap("RdYlGn").reversed()(x))
    return data


def format_data(data, divergent_value, **kwargs):
    data = data.apply(lambda x: "D" if x > divergent_value else x)
    if "sig_figs" in kwargs:
        notation = kwargs.get("notation", "standard")
        data = data.apply(
            lambda x: sigfig.round(x, kwargs["sig_figs"], notation=notation),
        )
    return data


def color_metrics(metric_df, color_df):
    # wrap the cell with the color
    latex_df = copy.deepcopy(metric_df)
    for i in range(latex_df.shape[0]):
        str_value = str(latex_df.iloc[i])
        color = color_df.iloc[i]
        color_str = "{}".format(",".join([f"{c:.2f}" for c in color[:3]]))
        latex_df.iloc[i] = r"\cellcolor[rgb]{" + color_str + "} " + str_value

    return latex_df


def color_data(
    df,
    column,
    min_value,
    mid_value,
    max_value,
    divergent_value,
    log,
    format,
):
    raw_df = copy.deepcopy(df[column])
    raw_df = raw_df.replace([np.inf, -np.inf], np.nan)
    raw_df = raw_df.apply(
        lambda x: divergent_value if x > divergent_value or np.isnan(x) else x,
    )

    # processed data used to compute colors
    if log:
        processed_df = raw_df.apply(lambda x: np.log10(x))
    else:
        processed_df = copy.deepcopy(raw_df)

    color_df = transform_to_colormap(processed_df, min_value, mid_value, max_value)

    # formatted raw data is used in the table
    formatted_raw_df = format_data(raw_df, divergent_value, **format)
    colored_metric_df = color_metrics(formatted_raw_df, color_df)
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


def main():
    directory = os.path.dirname(GravNN.__file__)
    df = pd.read_pickle(directory + "/../Data/Comparison/all_comparisons.data")
    # sort by percent_planes
    df = df.sort_values(by="percent_planes")
    # df = df.sort_values(by="percent_extrapolation")

    # ranking columns
    ranking_columns = [
        "percent_planes",
        "percent_interior",
        "percent_exterior",
        "percent_extrapolation",
        "percent_surface",
        "pos_error",
        "dt",
    ]

    # generate a rank for each model based on the ranking columns
    # then compute the average rank for each model
    # and sort the dataframe by average rank (the lower the number, the better the rank)
    df["rank"] = df[ranking_columns].rank().mean(axis=1)
    df = df.sort_values(by="rank")

    latex_df = copy.deepcopy(df)

    # conversions = [
    #     {
    #         "column": "percent_planes",
    #         "min_value": 0,
    #         "mid_value": 1,
    #         "max_value": None,
    #         "divergent_value": 100,
    #         "log": True,
    #         "format": {"sig_figs": 2},
    #     },
    #     {
    #         "column": "percent_interior",
    #         "min_value": 0,
    #         "mid_value": 1,
    #         "max_value": None,
    #         "divergent_value": 100,
    #         "log": True,
    #         "format": {"sig_figs": 2},
    #     },
    #     {
    #         "column": "percent_exterior",
    #         "min_value": 0,
    #         "mid_value": 1,
    #         "max_value": None,
    #         "divergent_value": 100,
    #         "log": True,
    #         "format": {"sig_figs": 2},
    #     },
    #     {
    #         "column": "percent_extrapolation",
    #         "min_value": 0,
    #         "mid_value": 3,
    #         "max_value": None,
    #         "divergent_value": 100,
    #         "log": False,
    #         "format": {"sig_figs": 2},
    #     },
    #     {
    #         "column": "percent_surface",
    #         "min_value": 0,
    #         "mid_value": 20,
    #         "max_value": None,
    #         "divergent_value": 100,
    #         "log": False,
    #         "format": {"sig_figs": 2},
    #     },
    #     {
    #         "column": "pos_error",
    #         "min_value": 3,
    #         "mid_value": None,
    #         "max_value": 8,
    #         "divergent_value": 1e8,
    #         "log": True,
    #         "format": {"sig_figs": 2, "notation": "scientific"},
    #     },
    #     {
    #         "column": "dt",
    #         "min_value": 1,
    #         "mid_value": None,
    #         "max_value": 30,
    #         "divergent_value": 100,
    #         "log": False,
    #         "format": {"sig_figs": 2},
    #     },
    #     {
    #         "column": "train_duration",
    #         "min_value": 0,
    #         "mid_value": 1,
    #         "max_value": None,
    #         "divergent_value": 600,
    #         "log": True,
    #         "format": {"sig_figs": 2},
    #     },
    #     {
    #         "column": "rank",
    #         "min_value": 9.7,
    #         "mid_value": None,
    #         "max_value": 55,
    #         "divergent_value": 56,
    #         "log": False,
    #         "format": {"sig_figs": 2},
    #     },
    # ]

    conversions = [
        {
            "column": "percent_planes",
            "min_value": df["percent_planes"].min(),
            "mid_value": df["percent_planes"][df["percent_planes"] < 100].median(),
            "max_value": None,
            "divergent_value": 100,
            "log": False,
            "format": {"sig_figs": 2},
        },
        {
            "column": "percent_interior",
            "min_value": df["percent_interior"].min(),
            "mid_value": df["percent_interior"][df["percent_interior"] < 100].median(),
            "max_value": None,
            "divergent_value": 100,
            "log": False,
            "format": {"sig_figs": 2},
        },
        {
            "column": "percent_exterior",
            "min_value": df["percent_exterior"].min(),
            "mid_value": df["percent_exterior"][df["percent_exterior"] < 100].median(),
            "max_value": None,
            "divergent_value": 100,
            "log": False,
            "format": {"sig_figs": 2},
        },
        {
            "column": "percent_extrapolation",
            "min_value": df["percent_extrapolation"].min(),
            "mid_value": df["percent_extrapolation"][
                df["percent_extrapolation"] < 100
            ].median(),
            "max_value": None,
            "divergent_value": 100,
            "log": False,
            "format": {"sig_figs": 2},
        },
        {
            "column": "percent_surface",
            "min_value": df["percent_surface"].min(),
            "mid_value": df["percent_surface"][df["percent_surface"] < 100].median(),
            "max_value": None,
            "divergent_value": 100,
            "log": False,
            "format": {"sig_figs": 2},
        },
        {
            "column": "pos_error",
            "min_value": df["pos_error"].min(),
            "mid_value": df["pos_error"][df["pos_error"] < 1e8].median(),
            "max_value": None,
            "divergent_value": 1e8,
            "log": False,
            "format": {"sig_figs": 2, "notation": "scientific"},
        },
        {
            "column": "dt",
            "min_value": df["dt"].min(),
            "mid_value": df["dt"][df["dt"] < 100].median(),
            "max_value": None,
            "divergent_value": 100,
            "log": False,
            "format": {"sig_figs": 2},
        },
        {
            "column": "train_duration",
            "min_value": df["train_duration"].min(),
            "mid_value": df["train_duration"][df["train_duration"] < 600].median(),
            "max_value": None,
            "divergent_value": 600,
            "log": False,
            "format": {"sig_figs": 2},
        },
        {
            "column": "rank",
            "min_value": df["rank"].min(),
            "mid_value": df["rank"][df["rank"] < 56].median(),
            "max_value": None,
            "divergent_value": 56,
            "log": False,
            "format": {"sig_figs": 2},
        },
    ]

    for conversion in conversions:
        column = conversion["column"]
        latex_df[column] = color_data(
            df,
            column,
            min_value=conversion["min_value"],
            mid_value=conversion["mid_value"],
            max_value=conversion["max_value"],
            divergent_value=conversion["divergent_value"],
            log=conversion["log"],
            format=conversion["format"],
        )
        print(column)

    # assign model_name as index
    latex_df = latex_df.set_index("model_name")
    # sort by name
    # latex_df = latex_df.sort_index()

    drop_columns = [
        "dt_a_single",
        "dt_a_batch",
        "layers",
        "shape",
        "elements",
        "deg",
        "state_error",
        "rms_planes",
        "num_units",
    ]
    latex_df = latex_df.drop(columns=drop_columns)

    # separate into acc_noise == 0.0 and 0.1
    latex_df_0 = latex_df[latex_df["acc_noise"] == 0.0]
    latex_df_1 = latex_df[latex_df["acc_noise"] == 0.1]

    latex_df_0_500 = latex_df_0[latex_df_0["N_train"] == 500]
    latex_df_0_50000 = latex_df_0[latex_df_0["N_train"] == 50000]

    latex_df_1_500 = latex_df_1[latex_df_1["N_train"] == 500]
    latex_df_1_50000 = latex_df_1[latex_df_1["N_train"] == 50000]

    save_data(latex_df, "table")
    save_data(latex_df_0, "table_0")
    save_data(latex_df_1, "table_1")
    save_data(latex_df_0_500, "table_0_500")
    save_data(latex_df_0_50000, "table_0_50000")
    save_data(latex_df_1_500, "table_1_500")
    save_data(latex_df_1_50000, "table_1_50000")


if __name__ == "__main__":
    main()
