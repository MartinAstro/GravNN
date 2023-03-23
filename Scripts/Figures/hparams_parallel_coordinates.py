import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sigfig

import GravNN


def concat_strings(values):
    new_list = []
    for value in values:
        new_list.append(["".join([f"{s}_" for s in value])[:-1]])
    return np.array(new_list).squeeze()


def make_column_numeric(df, column):
    # try converting object column to float
    try:
        df = df.astype({column: float})
        unique_strings = None

    except Exception:
        # if string, remove spaces + encode
        str_values = df[column].values
        str_values_concat = concat_strings(str_values)
        unique_strings = np.unique(str_values_concat)
        df.loc[:, column] = str_values_concat

        # If the column contains strings, make integers
        for i, string in enumerate(unique_strings):
            mask = df[column] == string
            df.loc[mask, column] = i + 1

        # convert type to float
        df = df.astype({column: float})

    return df, unique_strings


def scale_data(df, column):
    df, unique_strings = make_column_numeric(df, column)

    max_val = df[column].max()
    min_val = df[column].min()
    log_diff = np.log10(max_val) - np.log10(min_val)
    log_diff = 0.0 if np.isinf(log_diff) else log_diff

    prefix = ""
    values = df[column].values

    # Tick values default to all unique entries
    tick_values = []
    for value in np.unique(values):
        val_rounded = sigfig.round(value, sigfigs=3)
        tick_values.append(val_rounded)

    perturbations = np.zeros_like(values)
    PERT_FRAC = 0.02

    # Convert to log space if min-max delta too large
    MAX_LOG_DIFF = 1.0
    if log_diff >= MAX_LOG_DIFF and "mean" not in column:
        values = np.log10(values)
        tick_values = np.log10(tick_values)
        prefix = "log10 "

    # add noise to the data for enhanced visibility
    perturbations = []
    for value in values:
        if value == 0.0:
            value = 0.1
        pert = np.random.normal(0, np.abs(value * PERT_FRAC))
        perturbations.append(pert)

    # unless it's the results
    if "mean" in column:
        perturbations = np.zeros_like(values)

        # tick values can't be each unique entry
        # so divide into 8
        tick_values = []
        for value in np.linspace(min_val, max_val, 8):
            val_rounded = sigfig.round(value, sigfigs=3)
            tick_values.append(val_rounded)

    # clip results to sit within tick bounds
    min_tick = np.min(tick_values)
    max_tick = np.max(tick_values)
    values = np.clip(values, min_tick, max_tick)
    values += np.array(perturbations)
    return values, prefix, tick_values, unique_strings


def main():
    directory = os.path.dirname(GravNN.__file__)
    # df = pd.read_pickle(directory + "/../Data/Dataframes/test_metrics.data")
    # df = pd.read_pickle(directory + "/../Data/Dataframes/hparams_ll2_metrics.data")
    # df = pd.read_pickle(directory + "/../Data/Dataframes/sigma_search_metrics.data")
    df = pd.read_pickle(
        directory + "/../Data/Dataframes/fourier_experiment_032323_metrics.data",
    )

    percent_min = df["percent_mean"].min()
    percent_max = df["percent_mean"].mean() + df["percent_mean"].std() * 2
    metric_ticks = []
    for value in np.linspace(percent_min, percent_max, 8):
        value_rounded = sigfig.round(value, sigfigs=3)
        metric_ticks.append(value_rounded)

    name_dict = {
        "rms_mean": "RMS mean",
        "percent_mean": "Percent mean",
        # "epochs": "Epochs",
        # "loss_fcns": "Loss Functions",
        "N_train": "Training Data",
        "fourier_features": "Fourier Features",
        "fourier_sigma": "Fourier Sigma",
        "freq_decay": "Fourier Decay",
        "trainable": "Trainable FF",
        "shared_freq": "Shared Frequencies",
        "shared_offset": "Shared Offsets",
        # "N_val": "Validation Data",
        # "learning_rate": "Learning Rate",
        # "num_units": "Nodes per Layer",
        # "network_type": "Architecture",
        # "preprocessing": "Preprocessing",
        # "dropout": "Dropout",
        # "activation": "Activation",
    }
    df = df.rename(columns=name_dict)

    labels_dict = {
        "Percent mean": {
            "range": [percent_min, percent_max],
            "tickvals": metric_ticks,
        },
    }
    hparams_df = df[list(name_dict.values())]

    dimensions = []
    for column in hparams_df.columns:
        values, prefix, tick_values, unique_strings = scale_data(hparams_df, column)

        # update the label to be prettier
        column_dict = labels_dict.get(column, {})
        dimension_dict = {
            "label": prefix + column,
            "values": values,
            "tickvals": column_dict.get("tickvals", tick_values),
            "ticktext": unique_strings,
            "range": column_dict.get("range", None),
        }

        dimensions.append(dimension_dict)

    # Log projection : https://stackoverflow.com/questions/48421782/plotly-parallel-coordinates-plot-axis-styling

    fig = go.Figure(
        data=go.Parcoords(
            line=dict(
                color=hparams_df["Percent mean"],
                colorscale=px.colors.diverging.Tealrose,
                # cmid=7.6,
                cmax=percent_max,
                cmin=hparams_df["Percent mean"].min(),
            ),
            dimensions=dimensions,
        ),
    )

    DPI_factor = 3
    DPI = 100  # standard DPI for matplotlib
    fig.update_layout(
        # autosize=True,
        height=2.7 * DPI * DPI_factor,
        width=6.5 * DPI * DPI_factor,
        template="none",
        font={
            "family": "serif",
            "size": 20,  # *DPI_factor
        },
    )
    directory = os.path.dirname(GravNN.__file__)
    # write_image(
    #     fig,
    #     figure_path + "hparams.pdf",
    #     format="pdf",
    #     width=6.5 * DPI * DPI_factor,
    #     height=3 * DPI * DPI_factor,
    # )

    fig.show()


if __name__ == "__main__":
    main()
