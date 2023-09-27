import pandas as pd
from sigfig import round
from utils import get_color, remove_rule_lines


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
        [[df["N_train"]], [df["PINN_constraint_fcn"]]],
        names=["N", "Constraint"],
    )
    str_df = pd.DataFrame(data=[[value]], columns=["%" + str(percent * 100)]).set_index(
        index,
    )
    return str_df


def generate_table(df2, index, category, prefix=""):
    table_df = pd.DataFrame()
    for i in range(len(df2)):
        row = df2.iloc[i]
        str_row = convert_to_formatted_string(row, category)
        try:
            table_df = table_df.combine_first(str_row)
        except:
            table_df = table_df.append(str_row)

    caption = (
        category
        + " PINN Constraint Performance as listed by $\\bar{e} \pm \sigma (e_{\\text{max}})$ where $e$ represents the percent error of the acceleration vectors."
    )
    label = "tab:pinn_constraint_performance"
    column_format = "|c" * (len(table_df.columns) + len(index[0])) + "|"
    table = table_df.to_latex(caption=caption, label=label, column_format=column_format)
    table = remove_rule_lines(table)

    file_name = (
        "Notes/PINN_Asteroid_Journal/Assets/" + prefix + category + "_nn_table.tex"
    )
    with open(file_name, "w") as f:
        f.write(table)
    print(table)


def generate_all_tables(df, prefix):
    constraints_as_strings = []
    for i in range(len(df["PINN_constraint_fcn"])):
        fcn = df["PINN_constraint_fcn"].iloc[i]
        network_type = df["network_type"].iloc[i]
        name = fcn.__name__
        words = name.split("_")
        if "Transformer" in network_type.__name__:
            network_name = "Transformer"
        else:
            network_name = "PINN"
        if words[0].upper() == "NO" and network_name == "PINN":
            new_name = "PINN 00"
        elif words[0].upper() == "NO" and network_name == "Transformer":
            new_name = "Transformer 00"
        else:
            new_name = network_name + " " + words[1].upper()
        constraints_as_strings.append(new_name)
    df["PINN_constraint_fcn"] = constraints_as_strings

    # df = df[df['PINN_constraint_fcn'] != 'PINN AP']
    # df = df[df['PINN_constraint_fcn'] != 'PINN APLC']

    df = df.sort_values(
        by=["N_train", "PINN_constraint_fcn", "acc_noise"],
        ascending=[False, True, True],
    )
    index = pd.MultiIndex.from_product(
        [
            df["N_train"].unique(),
            df["PINN_constraint_fcn"].unique(),
            df["acc_noise"].unique(),
        ],
        names=["N", "Constraint", "Noise"],
    )
    df2 = df.set_index(index)

    # generate_table('surface')
    generate_table(df2, index, "surface", prefix)
    generate_table(df2, index, "interior", prefix)
    generate_table(df2, index, "exterior", prefix)


def main():
    df = pd.read_pickle("Data/Dataframes/eros_official_w_noise.data")
    prefix = "pinn_"
    generate_all_tables(df, prefix)

    df = pd.read_pickle(
        "Data/Dataframes/eros_official_noise_transformer_no_annealing.data",
    )
    prefix = "transformer_"
    generate_all_tables(df, prefix)

    df = pd.read_pickle(
        "Data/Dataframes/eros_official_noise_transformer_annealing.data",
    )
    prefix = "annealing_transformer_"

    df = pd.read_pickle("Data/Dataframes/eros_official_transformer_pinn_40.data")
    prefix = "pinn40_"

    generate_all_tables(df, prefix)


if __name__ == "__main__":
    main()
