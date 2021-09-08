
from os import remove
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sigfig import round


# convert dataframe into proper string formatting
def convert_to_formatted_string(df, category):
    mean_error = df.loc['mean_error']
    std_error = df.loc['std_error']
    max_error = df.loc['max_error']

    mean_not = 'sci' if mean_error > 1E2 else 'std'
    std_not = 'sci' if std_error > 1E2 else 'std'
    max_not = 'sci' if max_error > 1E2 else 'std'

    mean_sig = 1 if mean_error > 1E2 else 3
    std_sig = 1 if std_error > 1E2 else 3
    max_sig = 1 if max_error > 1E2 else 3

    mean_value = round(float(mean_error), sigfigs=mean_sig, notation=mean_not)
    std_value = round(float(std_error), sigfigs=std_sig, notation=std_not)
    max_value = round(float(max_error), sigfigs=max_sig, notation=max_not)

    mean_color = get_color(mean_error)
    std_color = get_color(std_error)
    max_color =  get_color(max_error)

    value = "%s{%s} pm %s{%s} (%s{%s})" % (mean_color, mean_value, std_color, std_value, max_color, max_value)
    percent = df.loc['noise']
    index = pd.MultiIndex.from_product([[int(df['N_data'])], [int(df['N'])]], names=['Samples', 'N'])

    str_df = pd.DataFrame(data=[[value]], columns=["%" + str(percent*100)]).set_index(index)
    return str_df


def remove_rule_lines(table):
    lines = table.splitlines(True)
    new_table = []
    for line in lines:
        if 'rule' in line:
            new_line = '\\hline\n'
        else:
            new_line = line
            new_line = new_line.replace('pm', '$\\pm$')
            new_line = new_line.replace('textcolor', '\\textcolor')
            new_line = new_line.replace('\{', '{')
            new_line = new_line.replace('\}', '}')
            new_line = new_line.replace('\$', '$')
            new_line = new_line.replace('rt', '\\rt')
            new_line = new_line.replace('bt', '\\bt')
            new_line = new_line.replace('gt', '\\gt')
            new_line = new_line.replace('yt', '\\yt')
            new_line = new_line.replace('lt', '\\lt')
            new_line = new_line.replace('mt', '\\mt')

        new_table.append(new_line)
    return "".join(new_table)

def get_color(value):
    if value >= 1000:
        color = r'mt'
    elif value <1000 and value >= 100:
        color = r'rt'
    elif value < 100 and value >= 10:
        color = r'yt'
    else:
        if value < 5:
            color = r'gt'
        else:
            color = r'lt'
    return color


def generate_table(df, category):
    df = df[df['M'] == 0]
    df = df.sort_values(by=['N_data', 'N', 'M', 'noise'], ascending=[False, False, True,True])
    index = pd.MultiIndex.from_product([df['N_data'].unique(), df['N'].unique(), df['M'].unique(), df['noise'].unique()], names=['N-train', 'N', 'M', 'Noise'])
    df2 = df.set_index(index)

    table_df = pd.DataFrame()
    for i in range(len(df2)):
        row = df2.iloc[i]
        str_row = convert_to_formatted_string(row, category)
        try:
            table_df = table_df.combine_first(str_row)
        except:
            table_df = table_df.append(str_row)

    caption=category+' SH Performance as listed by $\bar{e} \pm \sigma (e_{\text{max}})$ where $e$ represents the percent error of the acceleration vectors.'
    label='tab:pinn_constraint_performance'
    column_format = "|c"*(len(table_df.columns)+len(index[0])-1)+"|"
    table = table_df.to_latex(caption=caption, label=label, column_format=column_format)
    table = remove_rule_lines(table)
    print(table)

    file_name = "Notes/PINN_Asteroid_Journal/Assets/" + category + "_sh_table.tex"
    with open(file_name, 'w') as f:
        f.write(table)


def main():
    df = pd.read_pickle("Data/Dataframes/BLLS_r_outer_stats.data")
    generate_table(df, 'exterior')

    df = pd.read_pickle("Data/Dataframes/BLLS_r_inner_stats.data")
    generate_table(df, 'interior')

    df = pd.read_pickle("Data/Dataframes/BLLS_r_surface_stats.data")
    generate_table(df, 'surface')

if __name__ == '__main__':
    main()