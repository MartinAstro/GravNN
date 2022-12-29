import plotly.graph_objects as go
import pandas as pd
import os
import numpy as np
import GravNN
import plotly.express as px
from plotly.io import write_image
from sigfig import round

def concat_strings(values):
    new_list = []
    for value in values:
        new_list.append(["".join([f"{s}_" for s in value])[:-1]])
    return np.array(new_list).squeeze()

def make_column_numerics(df, column):
    try:
        df.loc[:, column] = df.loc[:, column].astype(float)
        unique_strings = None
    except:

        try:
            unique_strings = df[column].unique()
        except:
            df.loc[:,column] = concat_strings(df[column].values)
            unique_strings = df[column].unique()
        for i, string in enumerate(unique_strings):
            mask = df[column] == string
            df.loc[mask, column] = i+1
        df.loc[:, column] = df.loc[:, column].astype(float)
    return df, unique_strings

def scale_data(df, column):
    df, unique_strings = make_column_numerics(df, column)
    values = df[column].values.astype(float)
    tick_values = [round(value, sigfigs=3) for value in np.unique(df[column].values.astype(float)).tolist()]
    log_diff = np.log10(np.max(values)) - np.log10(np.min(values))
    log_diff = 0.0 if np.isinf(log_diff) else log_diff
    prefix = ""

    if log_diff >= 1.0 and "_mean" not in column:
        values = np.log10(values)
        tick_values = np.log10(tick_values)
        prefix = "log10 "    
        # random must accept positive value in std (- log values don't work)
        values += np.array([np.random.normal(0, np.abs(value*0.02)) for value in values])
    elif "_mean" not in column: # add noise to columns that aren't the results 
        values += np.array([np.random.normal(0, np.max(values)*0.02) for value in values])
    else:
        min_result = np.min(values)
        max_result = np.max(values)
        tick_values = [round(value, sigfigs=3) for value in np.linspace(min_result, max_result, 8).tolist()]
    
    values = np.clip(values, a_min=np.min(tick_values), a_max=np.max(tick_values))

    return values, prefix, tick_values, unique_strings

def main():

    directory = os.path.dirname(GravNN.__file__)
    # df = pd.read_pickle(directory + "/../Data/Dataframes/test_metrics.data")
    df = pd.read_pickle(directory + "/../Data/Dataframes/multiFF_hparams_metrics.data")


    percent_min = df['percent_mean'].min()
    percent_max = df['percent_mean'].mean() + df['percent_mean'].std()*2
    labels_dict={
        "rms_mean": {
            "label" :"RMS",
            },
        "percent_mean": {
            "label" :"Percent",
            "range" : [percent_min, percent_max],
            "tickvals" : [round(value, sigfigs=3) for value in np.linspace(percent_min, percent_max, 8).tolist()]
            }, 
        # "magnitude_mean": {
        #     "label" :"Magnitude",
        #     },
        # "angle_mean": {
        #     "label" :"Angle",
        #     }, 
        "epochs" : {
            "label" :"Epochs",
            }, 
        "loss_fcns" : {
            "label" :"Loss Functions",
            }, 
        "N_train" : {
            "label" :"Training Data",
            },
        "fourier_features" : {
            "label" :"Fourier Features",
            },
        "fourier_sigma" : {
            "label" :"Fourier Sigma",
            },
        "freq_decay" : {
            "label" :"Fourier Decay",
            },
        "N_val" : {
            "label" :"Validation Data",
            },
        "learning_rate" : {
            "label" :"Learning Rate",
            },
        "num_units" : {
            "label" :"Nodes per Layer",
            },
        "network_type" : {
            "label" :"Architecture",
            },
        "preprocessing" : {
            "label" :"Preprocessing",
            },
        "dropout" : {
            "label" :"dropout",
            },
        }
    hparams_df = df[labels_dict.keys()]

    dimensions = []
    for column in hparams_df.columns:
        values, prefix, tick_values, unique_strings = scale_data(hparams_df, column)
        
        # update the label to be prettier
        column_dict = labels_dict[column]
        label = column_dict['label']
        dimension_dict = {
            'label' : prefix + label,
            'values' : values,
            'tickvals' : column_dict.get('tickvals', tick_values),
            "ticktext" : unique_strings,
            'range' : column_dict.get('range', None)
        }
        
        dimensions.append(dimension_dict)

    # Log projection : https://stackoverflow.com/questions/48421782/plotly-parallel-coordinates-plot-axis-styling

    fig = go.Figure(data=go.Parcoords(line=dict(
            color=hparams_df['percent_mean'],
            colorscale=px.colors.diverging.Tealrose,
            # cmid=7.6,
            cmax=0.25,
            cmin=hparams_df['percent_mean'].min(),
            ),
        dimensions=dimensions
    ))


    DPI_factor = 2
    DPI = 100 # standard DPI for matplotlib
    fig.update_layout(
        # autosize=True,
        height=2.7*DPI*DPI_factor,
        width=6.5*DPI*DPI_factor,
        template='none',
        font={
            'family' : 'serif',
            'size' : 10*DPI_factor 
        })
    directory = os.path.dirname(GravNN.__file__)
    figure_path = directory + "/../Plots/"
    # write_image(fig, figure_path + "hparams.pdf", format='pdf', width=6.5*DPI*DPI_factor, height=3*DPI*DPI_factor)

    fig.show()

if __name__ == "__main__":
    main()
