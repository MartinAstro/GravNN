import pandas as pd
from sigfig import round
from utils import get_color


# convert dataframe into proper string formatting
def convert_to_formatted_string(df, category):
    mean_error = df.loc[category + "_percent_mean"]
    std_error = df.loc[category + "_percent_std"]
    max_error = df.loc[category + "_percent_max"]

    mean_not = "sci" if mean_error > 1e2 else "std"
    std_not = "sci" if std_error > 1e2 else "std"
    max_not = "sci" if max_error > 1e2 else "std"

    mean_sig = 1 if mean_error > 1e2 else 3
    std_sig = 1 if std_error > 1e2 else 3
    max_sig = 1 if max_error > 1e2 else 3

    mean_value = round(float(mean_error), sigfigs=mean_sig, notation=mean_not)
    std_value = round(float(std_error), sigfigs=std_sig, notation=std_not)
    max_value = round(float(max_error), sigfigs=max_sig, notation=max_not)

    mean_color = get_color(mean_error)
    std_color = get_color(std_error)
    max_color = get_color(max_error)

    value = "%s{%s} pm %s{%s} (%s{%s})" % (
        mean_color,
        mean_value,
        std_color,
        std_value,
        max_color,
        max_value,
    )

    percent = df.loc["acc_noise"]
    index = pd.MultiIndex.from_product(
        [
            [df["N_train"]],
            [df["PINN_constraint_fcn"]],
        ],
        names=["N", "Constraint"],
    )
    str_df = pd.DataFrame(
        data=[[value]],
        columns=["%" + str(percent * 100)],
    ).set_index(index)
    return str_df


def generate_table(df, df_40):
    params_40 = df_40.params.values
    avg_error_40 = df_40.percent_mean.values

    params = df.params.values
    N_FF = df.fourier_features.values
    avg_error = df.percent_mean.values

    table_df = pd.DataFrame(
        {
            "Parameters": params,
            "N_{FF}": N_FF,
            r"Avg. \% Error": avg_error,
            r"Rel. Model Size \% ": (params - params_40) / params_40 * 100,
            r"Avg. \% Error \% ": (avg_error - avg_error_40) / avg_error_40 * 100,
        },
    )

    output = table_df.to_latex()
    return output
    # generate_table(df2, index, "surface", prefix)


def main():
    df_file = "Data/Dataframes/network_40_metrics.data"
    df_40 = pd.read_pickle(df_file)

    df_file = "Data/Dataframes/fourier_experiment_not_trainable_032523_metrics.data"
    df = pd.read_pickle(df_file)
    not_trainable_FF = generate_table(df, df_40)

    df_file = "Data/Dataframes/fourier_experiment_trainable_032523_metrics.data"
    df = pd.read_pickle(df_file)
    trainable_FF = generate_table(df, df_40)

    df_file = "Data/Dataframes/fourier_experiment_trainable_shared_032523_metrics.data"
    df = pd.read_pickle(df_file)
    trainable_FF_shared = generate_table(df, df_40)

    print(not_trainable_FF)
    print(trainable_FF)
    print(trainable_FF_shared)


if __name__ == "__main__":
    main()
