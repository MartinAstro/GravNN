import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import GravNN

GRAY_RGBA = plt.get_cmap("gray")(0.9)
RED_RGBA = plt.get_cmap("RdYlGn").reversed()(1.0)


def get_color(value, min_val, max_val):
    # Non-numeric value, return red color
    SCALE_CONST = 0.2

    norm = plt.Normalize(min_val, max_val)
    cmap = plt.get_cmap("RdYlGn").reversed()  # Red to Green colormap

    # If unable to convert value to number, make red
    try:
        value = float(value)
    except Exception:
        return "{}".format(",".join([f"{c:.2f}" for c in RED_RGBA[:3]]))

    # If the value is NaN, return gray
    if np.isnan(value):
        return "{}".format(",".join([f"{c:.2f}" for c in GRAY_RGBA[:3]]))

    # otherwise, normalize the color with respect to the min and max values
    value_norm = norm(value)

    # reduce the range of colors to be between 0.3 and 0.7
    def scale(x):
        return x * (1 - 2 * SCALE_CONST) + SCALE_CONST

    lighter_value_norm = scale(value_norm)

    rgba = cmap(lighter_value_norm)

    # Convert RGBA to LaTeX compatible RGB format (0-1 scale)
    return "{}".format(",".join([f"{c:.2f}" for c in rgba[:3]]))


def get_bold_df(df):
    bold_df = pd.DataFrame(index=df.index, columns=df.columns)
    bold_df = bold_df.fillna(False)

    for col in df.columns:
        if col == "model_name":
            bold_df[col] = df[col]
            continue

        # Convert column to float
        float_col = pd.to_numeric(df[col], errors="coerce")
        min_val = np.nanmin(float_col)

        # change the bold_df value to True if equal to min_val
        bold_df[col] = float_col.apply(lambda x: x == min_val)
    return bold_df


def apply_color_per_column(df):
    colored_df = pd.DataFrame(index=df.index, columns=df.columns)

    # Standard color columns that range from 0 - 100
    standard_columns = [
        "percent_interior",
        "percent_exterior",
        "percent_extrapolation",
        "percent_planes",
        "percent_surface",
    ]
    # get max value from the standard columns excluding any strings or nans
    max_std_val = df[standard_columns].apply(pd.to_numeric, errors="coerce").max().max()
    print(max_std_val)

    for std_column in standard_columns:
        colored_df[std_column] = df[std_column].apply(
            lambda x: get_color(x, 0, max_std_val),
        )

    for col in df.columns:
        if col in standard_columns:
            continue
        if col == "model_name":
            colored_df[col] = df[col]
            continue
        # Convert column to float
        float_col = pd.to_numeric(df[col], errors="coerce")

        # Calculate IQR
        Q1 = float_col.quantile(0.25)
        Q3 = float_col.quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds to identify outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        min_val = float_col.min()
        max_val = float_col.max()

        # Check if range exceeds two orders of magnitude
        if max_val / min_val > 100:
            # Exclude outliers
            filtered_col = float_col[
                (float_col >= lower_bound) & (float_col <= upper_bound)
            ]
            max_val = filtered_col.max() if not filtered_col.empty else max_val
            min_val = filtered_col.min() if not filtered_col.empty else min_val

        colored_df[col] = float_col.apply(lambda x: get_color(x, min_val, max_val))
    return colored_df


def shorten_sci_notation(value):
    str_value = str(value)
    if "e+0" in str_value:
        str_value = str_value.replace("e+0", "e")
    return str_value


def replace_nan_w_na(value):
    str_value = str(value)
    if "nan" in str_value:
        str_value = str_value.replace("nan", "NA")
    return str_value


def apply_bold(value, bold):
    if bold:
        if value == "nan":
            return value
        return "\\textbf{" + value + "}"
    else:
        return value


def df_to_latex(df, df_colored, df_bold):
    latex_df = df.copy()
    for i in range(len(df)):
        for j in range(len(df.columns)):
            original_value = df.iloc[i, j]
            is_bold = df_bold.iloc[i, j]
            str_value = shorten_sci_notation(original_value)
            str_value = apply_bold(str_value, is_bold)
            str_value = replace_nan_w_na(str_value)
            color_str = df_colored.iloc[i, j]

            # If the value was never populated, make it gray
            if isinstance(color_str, float):
                if np.isnan(color_str):
                    color_str = "{}".format(
                        ",".join([f"{c:.2f}" for c in GRAY_RGBA[:3]]),
                    )

            latex_df.iloc[i, j] = r"\cellcolor[rgb]{" + color_str + "} " + str_value

    column_format = "r" + "l" * len(df.columns)
    latex_str = latex_df.to_latex(
        escape=False,
        column_format=column_format,
        multicolumn_format="c",
        multirow=True,
        multicolumn=True,
        bold_rows=False,
    )
    return latex_str


def rotate_headers(latex_str, angle=45):
    lines = latex_str.splitlines()
    new_lines = []
    for line in lines:
        if "Interior" in line:
            # if ' & ' in line and '\\toprule' not in line and '\\midrule' not in line and '\\bottomrule' not in line:
            # Split the line at ' & ' and wrap each header with the rotatebox command
            headers = line.split(" & ")
            headers_rotated = []
            for header in headers:
                # check if string is just white spaces
                if header.strip() == "{}":
                    headers_rotated.append(header)
                else:
                    header_stripped = header.strip()
                    ending_line = False
                    if header_stripped[-2:] == "\\\\":
                        ending_line = True
                        header_stripped = header_stripped[:-2]
                    if " " in header_stripped:
                        header_stripped = header_stripped.replace(" ", "\\\\ ")
                        header_stripped = (
                            r"\begin{tabular}{@{}c@{}}"
                            + header_stripped
                            + r"\end{tabular}"
                        )
                    new_cell = (
                        "\\rotatebox[origin=l]{"
                        + str(angle)
                        + "}{"
                        + header_stripped
                        + "}"
                    )
                    if ending_line:
                        new_cell += "\\\\"
                    headers_rotated.append(new_cell)

            line = " & ".join(headers_rotated)
            line += "\\\\"
        new_lines.append(line)
    return "\n".join(new_lines)


def to_sci(x):
    try:
        ge1000 = x >= 1000
        le01 = x <= 0.01
        notZero = x != 0
        if (ge1000 or le01) and notZero:
            y = "{:.1e}".format(x)
        else:
            y = x
        return y
    except Exception:
        return x


def mark_diverged(x):
    ge100 = x >= 100
    if ge100:
        y = "D"
    else:
        y = x
    return y


def update_index(df):
    masked_df = df.reset_index()
    masked_df = masked_df.set_index("model_name")
    masked_df.index.name = None

    # Remove underscores from index strings
    masked_df.index = masked_df.index.str.replace("_", " ")

    # Remove Small and Large Suffixes
    masked_df.index = masked_df.index.str.replace("Small", "")
    masked_df.index = masked_df.index.str.replace("Large", "")

    # Sort index in a specified order (i.e. PM, SH, Poly, etc)
    by_values = [
        "PM",
        "SH",
        "POLYHEDRAL",
        "MASCONS",
        "ELM",
        "TNN ",
        "PINN I ",
        "PINN II ",
        "PINN III ",
    ]
    masked_df = masked_df.reindex(by_values)

    # Rename Polyhedral
    masked_df = masked_df.rename(
        index={
            "POLYHEDRAL": "Poly.",
            "MASCONS": "Mascons",
            "TNN": "NN",
        },
    )

    return masked_df


def process_df(df, primary_query):
    ###################
    # PREPROCESSING
    ###################

    # Unwrap any wrapped variables
    if isinstance(df.model_name[0], list):
        df.model_name = df.model_name.apply(lambda x: x[0])
    if isinstance(df.N_train[0], list):
        df.N_train = df.N_train.apply(lambda x: x[0])
    if isinstance(df.acc_noise[0], list):
        df.acc_noise = df.acc_noise.apply(lambda x: x[0])

    # Add a new column that categorizes the model into Big or Small
    df["size"] = df["num_params"].apply(lambda x: "Small" if x < 10000 else "Large")

    # make position error in km
    df["pos_error"] = df["pos_error"] / 1000

    # make queriable idx
    df["idx"] = df.index

    # Set data precision
    df = df.round(1)

    # Mark diverged value for only some columns
    df["percent_surface"] = df["percent_surface"].apply(mark_diverged)
    df["percent_interior"] = df["percent_interior"].apply(mark_diverged)
    df["percent_exterior"] = df["percent_exterior"].apply(mark_diverged)
    df["percent_planes"] = df["percent_planes"].apply(mark_diverged)
    df["percent_extrapolation"] = df["percent_extrapolation"].apply(mark_diverged)

    # I need to duplicate the PM model so they also get selected into the size == 'Large' queries
    # I need to duplicate Polyhedral so they also get selected into the N_train == 50000 queries
    query = "model_name == 'PM'"
    PM_rows = copy.deepcopy(df.query(query))
    PM_rows["size"] = "Large"
    PM_rows["idx"] = len(df) + np.arange(0, len(PM_rows["idx"]), 1, dtype=int)
    df = pd.concat([df, PM_rows])

    query = "model_name == 'POLYHEDRAL'"
    poly_rows = copy.deepcopy(df.query(query))
    poly_rows["N_train"] = 50000
    idx_col = np.arange(0, len(poly_rows["idx"]), 1, dtype=int)
    poly_rows["idx"] = len(df) + idx_col
    df = pd.concat([df, poly_rows])

    df = df.set_index("idx")

    # Set polyhedral train duration to NaN
    df.loc[
        df["model_name"] == "POLYHEDRAL",
        "train_duration",
    ] = np.nan

    # The polyhedral is not applicable for the 0.1 noise case
    if "acc_noise == 0.1" in primary_query:
        df.loc[df["model_name"] == "POLYHEDRAL", :] = np.nan

    # The PM is not applicable for the Large size case
    if "size == 'Large'" in primary_query:
        df.loc[df["model_name"] == "PM", :] = np.nan

    return df


def produce_table(
    noise,
    size,
    N_train,
    include_header_row=False,
    include_footer_row=False,
):
    directory = os.path.dirname(GravNN.__file__)
    # df = pd.read_pickle(directory + "/../Data/Dataframes/comparison_metrics.data")
    # df = pd.read_pickle(directory + "/../Data/Comparison/old_comparison/all_comparisons.data")
    df = pd.read_pickle(directory + "/../Data/Comparison/all_comparisons.data")

    # query for noise, size, and training data
    query = f"acc_noise == {noise} and size == {size} and N_train == {N_train}"

    processed_df = process_df(df, query)

    # Use the query to make a mask that can applied to other dataframes
    # without using the index
    mask = processed_df.query(query).index

    # Convert columns to scientific notation
    processed_df = processed_df.applymap(to_sci)

    # Color the dataframe
    df_colored = apply_color_per_column(processed_df)

    masked_df = processed_df.iloc[mask]
    colors_df = df_colored.iloc[mask]
    bold_df = get_bold_df(masked_df)

    masked_df = update_index(masked_df)
    colors_df = update_index(colors_df)
    bold_df = update_index(bold_df)

    # Drop unused columns
    column_rename = {
        "percent_planes": ("Planes", "Error[\%]"),
        "percent_extrapolation": ("Generalization", "Extrapol. Error[\%]"),
        "percent_exterior": ("Generalization", "Exterior Error[\%]"),
        "percent_interior": ("Generalization", "Interior Error[\%]"),
        "percent_surface": ("Surface", "Error[\%]"),
        "pos_error": ("Trajectory", "Position Error[km]"),
        "dt": ("Trajectory", "Propagation Time[s]"),
        # "dt_a_single": ("Trajectory", "Evaluation Time[s]"),
        # "size": ('Training', 'Size'),
        "N_train": ("Auxillary", "Samples"),
        "acc_noise": ("Auxillary", "Noise[\%]"),
        "num_params": ("Auxillary", "Params"),
        "train_duration": ("Auxillary", "Regression Time[s]"),
    }

    # Reorder columns
    masked_df = masked_df[list(column_rename.keys())]
    colors_df = colors_df[list(column_rename.keys())]
    bold_df = bold_df[list(column_rename.keys())]

    # Rename / Group some of the columns via MultiIndex
    masked_df.columns = pd.MultiIndex.from_tuples(list(column_rename.values()))
    colors_df.columns = pd.MultiIndex.from_tuples(list(column_rename.values()))
    bold_df.columns = pd.MultiIndex.from_tuples(list(column_rename.values()))

    drop_columns = [("Auxillary", "Noise[\%]"), ("Auxillary", "Samples")]
    processed_df = masked_df.drop(columns=drop_columns)
    df_colored = colors_df.drop(columns=drop_columns)
    df_bold = bold_df.drop(columns=drop_columns)

    # Wrap a red to green color cell mapping around the data
    latex_table = df_to_latex(processed_df, df_colored, df_bold)
    latex_table = rotate_headers(latex_table, angle=65)

    header = f"N = {N_train} Samples; \t Noise = {noise} \%; \t Model Size = {size}"
    header = f"N = {N_train} Samples; \quad Model Size = {size}"
    new_header_col_len = len(column_rename) - len(drop_columns) + 1
    new_header = (
        "\\multicolumn{" + str(new_header_col_len) + "}{c}{" + header + "} \\\\"
    )
    empty_line = "\\multicolumn{" + str(new_header_col_len) + "}{c}{} \\\\"
    if not include_header_row:
        # keep the 0 row, skip the 1,2,3,4 rows
        latex_table = "\n".join(latex_table.split("\n")[5:])

    latex_table = "\n".join(
        latex_table.split("\n")[:-2]
        + [empty_line]
        + [new_header]
        + latex_table.split("\n")[-2:],
    )

    if not include_footer_row:
        latex_table = "\n".join(latex_table.split("\n")[:-1] + [empty_line])

    print(latex_table)
    return latex_table


def main():
    def save_table(acc_noise):
        latex_table = "\n"
        latex_table += produce_table(
            acc_noise,
            "'Small'",
            50000,
            include_header_row=True,
        )
        latex_table += produce_table(
            acc_noise,
            "'Large'",
            50000,
            include_footer_row=False,
        )
        latex_table += produce_table(
            acc_noise,
            "'Small'",
            500,
            include_footer_row=False,
        )
        latex_table += produce_table(acc_noise, "'Large'", 500, include_footer_row=True)

        # Get home directory
        home_dir = os.path.expanduser("~")
        path = f"{home_dir}/Documents/Research/Papers/Journal/PINN_III/Snippets/table_{acc_noise}.tex"
        with open(path, "w") as f:
            f.write(latex_table)

    save_table(0)
    save_table(0.1)


if __name__ == "__main__":
    main()
