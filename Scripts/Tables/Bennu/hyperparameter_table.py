# %%
import pandas as pd
import numpy as np
import copy


trad_df = pd.read_pickle( "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\Data\\Dataframes\\traditional_nn_df.data").sort_values(by='params')
#pinn_df = pd.read_pickle( "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\Data\\Dataframes\\pinn_df.data")

df = trad_df
# # Non-PINN


# %%
df.T
df.columns
sub_df = df[['PINN_flag', 'num_units', 'params', 'learning_rate', 'batch_size', 'optimizer', 'initializer', 'epochs', 'decay_epoch_0', 'decay_rate', 'decay_rate_epoch', 'x_transformer', 'a_transformer']]

# %%
labels = []
pinn_flag = copy.deepcopy(sub_df['PINN_flag'].values)
num_units = copy.deepcopy(sub_df['num_units'].values)


print(pinn_flag)
for i in range(len(sub_df)):
    if pinn_flag[i] == 'none':
        pinn_flag[i] = 'Traditional'
    else:
        pinn_flag[i] = 'PINN'
    labels.append(pinn_flag[i] + " " + str(num_units[i]))

labels
sub_df.index=labels
sub_df.columns


# %%
table_df = sub_df[['params', 'learning_rate', 'batch_size',
       'optimizer', 'initializer', 'epochs', 'decay_epoch_0', 'decay_rate',
       'decay_rate_epoch', 'x_transformer', 'a_transformer']].T

caption='Hyperparameters for the traditional and physics-informed neural networks trained in this paper.'
label='tab:parameters'
column_format = "|c"*len(table_df.columns)+"|"

table_df    


# %%
print(table_df.to_latex(caption=caption, label=label, column_format=column_format))


