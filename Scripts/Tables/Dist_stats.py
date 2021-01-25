import pandas as pd
import numpy as np

def add_stats_columns(sub_df):
    brill_sh = sub_df['Brillouin_sh_rse_mean']
    sub_df['dC_Brill'] = (sub_df['params'] - brill_sh*(brill_sh+1))/(brill_sh*(brill_sh+1))

    brill_F_sh = sub_df['Brillouin_sh_sigma_2_mean']
    sub_df['dC_F_Brill'] = (sub_df['params'] - brill_F_sh*(brill_F_sh+1))/(brill_F_sh*(brill_F_sh+1))


    LEO_sh = sub_df['LEO_sh_rse_mean']
    sub_df['dC_LEO'] = (sub_df['params'] - LEO_sh*(LEO_sh+1))/(LEO_sh*(LEO_sh+1))

    LEO_F_sh = sub_df['LEO_sh_sigma_2_mean']
    sub_df['dC_F_LEO'] = (sub_df['params'] - LEO_F_sh*(LEO_F_sh+1))/(LEO_F_sh*(LEO_F_sh+1))

    return sub_df


def main():
    rand_df = pd.read_pickle( "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\N_1000000_Rand_Study.data").sort_values(by='params')
    exp_df = pd.read_pickle( "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\N_1000000_exp_norm_study.data")

    # # Non-PINN

    sub_df = rand_df
    sub_df = add_stats_columns(sub_df)
    caption='Random Distribution Statistics'
    label='tab:RandDistStats'
    print(sub_df.to_latex(columns=['layers', 'dC_Brill', 'dC_F_Brill', 'dC_LEO', 'dC_F_LEO'],float_format="%.2f",index=False,caption=caption, label=label, column_format='|c|c|c|c|c|'))

    sub_df = exp_df[pd.isna(exp_df['mu'])][exp_df['invert'] == True][exp_df['scale_parameter']==140000.0].sort_values(by='params')
    sub_df = add_stats_columns(sub_df)
    caption='$\mathbb{E(x, 420, 14)} Distribution Statistics'
    label='tab:E_420_14_stats'
    print(sub_df.to_latex(columns=['layers', 'dC_Brill', 'dC_F_Brill', 'dC_LEO', 'dC_F_LEO'],float_format="%.2f",index=False,caption=caption, label=label, column_format='|c|c|c|c|c|'))


    sub_df = exp_df[pd.isna(exp_df['mu'])][exp_df['invert'] == False][exp_df['scale_parameter']==140000.0].sort_values(by='params')
    sub_df = add_stats_columns(sub_df)
    caption='$\mathbb{E(x, 0, 14)} Distribution Statistics'
    label='tab:E_0_14_stats'
    print(sub_df.to_latex(columns=['layers', 'dC_Brill', 'dC_F_Brill', 'dC_LEO', 'dC_F_LEO'],float_format="%.2f",index=False,caption=caption, label=label, column_format='|c|c|c|c|c|'))


    sub_df = exp_df[pd.isna(exp_df['mu'])][exp_df['invert'] == True][exp_df['scale_parameter']==42000.0].sort_values(by='params')
    sub_df = add_stats_columns(sub_df)
    caption='$\mathbb{E(x, 420, 42)} Distribution Statistics'
    label='tab:E_420_42_stats'
    print(sub_df.to_latex(columns=['layers', 'dC_Brill', 'dC_F_Brill', 'dC_LEO', 'dC_F_LEO'],float_format="%.2f",index=False,caption=caption, label=label, column_format='|c|c|c|c|c|'))


    sub_df = exp_df[pd.isna(exp_df['mu'])][exp_df['invert'] == False][exp_df['scale_parameter']==42000.0].sort_values(by='params')
    sub_df = add_stats_columns(sub_df)
    caption='$\mathbb{E(x, 0, 42)} Distribution Statistics'
    label='tab:E_0_42_stats'
    print(sub_df.to_latex(columns=['layers', 'dC_Brill', 'dC_F_Brill', 'dC_LEO', 'dC_F_LEO'],float_format="%.2f",index=False,caption=caption, label=label, column_format='|c|c|c|c|c|'))

    # PINN
    rand_df = pd.read_pickle( "C:\\Users\\John\\Documents\\Research\\ML_Gravity\\N_1000000_PINN_Study.data").sort_values(by='params')
    sub_df = rand_df
    sub_df = add_stats_columns(sub_df)
    caption='Random Distribution PINN Statistics'
    label='tab:RandDistStats'
    print(sub_df.to_latex(columns=['layers', 'dC_Brill', 'dC_F_Brill', 'dC_LEO', 'dC_F_LEO'],float_format="%.2f",index=False,caption=caption, label=label, column_format='|c|c|c|c|c|'))


if __name__ == '__main__':
    main()


