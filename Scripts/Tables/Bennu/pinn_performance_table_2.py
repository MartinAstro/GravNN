
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

df = pd.read_pickle("Data/Dataframes/bennu_official_w_noise_2.data")


constraints_as_strings = []
for fcn in df['PINN_constraint_fcn']:
    name = fcn.__name__
    words = name.split("_")
    new_name = words[0].upper() + " " + words[1].upper()
    constraints_as_strings.append(new_name)
df['PINN_constraint_fcn'] = constraints_as_strings

df = df.sort_values(by=['N_train', 'PINN_constraint_fcn', 'acc_noise'], ascending=[False, True,True])
index = pd.MultiIndex.from_product([df['N_train'].unique(), df['PINN_constraint_fcn'].unique(), df['acc_noise'].unique()], names=['N', 'Constraint', 'Noise'])
index
df2 = df.set_index(index)


# convert dataframe into proper string formatting
def convert_to_formatted_string(df, category):
    value = "%.2f pm %.2f (%.2f)" % (df.loc[category+'_percent_mean'], df.loc[category+'_percent_std'], df.loc[category+'_percent_max'])
    percent = df.loc['acc_noise']
    index = pd.MultiIndex.from_product([[df['N_train']], [df['PINN_constraint_fcn']]], names=['N', 'Constraint'])
    str_df = pd.DataFrame(data=[[value]], columns=["\%" + str(percent*100)]).set_index(index)
    return str_df

def generate_table(category):
    table_df = pd.DataFrame()
    for i in range(len(df2)):
        row = df2.iloc[i]
        str_row = convert_to_formatted_string(row, category)
        try:
            table_df = table_df.combine_first(str_row)
        except:
            table_df = table_df.append(str_row)

    caption=category+' Performance.'
    label='tab:pinn_constraint_performance'
    column_format = "|c"*len(table_df.columns)+"|"

    print(table_df.to_latex(caption=caption, label=label, column_format=column_format))

# generate_table('surface')
generate_table('interior')
generate_table('exterior')

