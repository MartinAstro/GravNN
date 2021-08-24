import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
# df = pd.read_pickle( "Data/Dataframes/medium_data_pinn_constraints_wo_annealing_lr_plateau.data")
# df = df.append(pd.read_pickle( "Data/Dataframes/small_data_pinn_constraints_wo_annealing.data"))
# df = df.append(pd.read_pickle( "Data/Dataframes/tiny_data_pinn_constraints_wo_annealing_lr_plateau.data"))
# df = df.append(pd.read_pickle( "Data/Dataframes/v_tiny_data_pinn_constraints_wo_annealing_lr_plateau.data"))
# df = df.append(pd.read_pickle( "Data/Dataframes/v_v_tiny_data_pinn_constraints_wo_annealing_lr_plateau.data"))
# df = df.append(pd.read_pickle( "Data/Dataframes/no_pinn.data"))

df = pd.read_pickle("Data/Dataframes/transformer_wo_annealing.data")

# convert dataframe into proper string formatting
def convert_to_formatted_string(df):
    surface = r"%.2f \pm %.2f (%.2f)"  % (df.loc['surface_percent_mean'], df.loc['surface_percent_std'], df.loc['surface_percent_max'])
    interior = r"%.2f \pm %.2f (%.2f)" % (df.loc['interior_percent_mean'], df.loc['interior_percent_std'], df.loc['interior_percent_max'])
    exterior = r"%.2f \pm %.2f (%.2f)" % (df.loc['exterior_percent_mean'], df.loc['exterior_percent_std'], df.loc['exterior_percent_max'])
    str_df = pd.DataFrame(data=[[df.loc["N_train"], df.loc['PINN_constraint_fcn'].__name__, surface, interior, exterior]], columns=['N_train', 'Constraint', 'Surface', 'Interior', 'Exterior']).set_index('N_train')
    return str_df

table_df = pd.DataFrame()
plt.figure()
for i in range(len(df)):
    row = df.iloc[i]
    str_row = convert_to_formatted_string(row)
    table_df = table_df.append(str_row)
    plt.plot(row['history']['val_loss'], label="%s %.2f" % (row['PINN_constraint_fcn'].__name__, row.loc['exterior_percent_mean']))


plt.legend()
plt.show()
caption='PINN Constraint Performance.'
label='tab:pinn_constraint_performance'
column_format = "|c"*len(table_df.columns)+"|"

print(table_df.to_latex(caption=caption, label=label, column_format=column_format))
