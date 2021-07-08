
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_vlim_bounds(dist, sigma):
    mu = np.mean(dist)
    std = np.std(dist)
    vlim_min = clamp(mu-sigma*std, 0, np.inf)
    vlim_max = mu+sigma*std
    return [vlim_min, vlim_max]

def clamp(x, smallest, largest): 
    return max(smallest, min(x, largest))

def format_potential_as_Nx3(u):
    U_Nx3 = np.zeros((len(u), 3))
    try:
        U_Nx3[:,0] = u[:,0]
    except:
        U_Nx3[:,0] = u
    return U_Nx3

def sh_pareto_curve(file_name, max_deg=None, log=True, sigma=2):
    if max_deg is not None:
        sh_df = pd.read_pickle(file_name).loc[:max_deg]
    else:
        sh_df = pd.read_pickle(file_name)

    if log: 
        plt.semilogx(sh_df.index*(sh_df.index+1), sh_df['rse_mean'], label=r'MRSE($\mathcal{A}$)')
        plt.semilogx(sh_df.index*(sh_df.index+1), sh_df['sigma_'+str(sigma)+'_mean'], label=r'MRSE($\mathcal{F}$)')
        plt.semilogx(sh_df.index*(sh_df.index+1), sh_df['sigma_'+str(sigma)+'_c_mean'], label=r'MRSE($\mathcal{C}$)')
    else:
        plt.plot(sh_df.index*(sh_df.index+1), sh_df['rse_mean'], label=r'MRSE($\mathcal{A}$)')
        plt.plot(sh_df.index*(sh_df.index+1), sh_df['sigma_'+str(sigma)+'_mean'], label=r'MRSE($\mathcal{F}$)')
        plt.plot(sh_df.index*(sh_df.index+1), sh_df['sigma_'+str(sigma)+'_c_mean'], label=r'MRSE($\mathcal{C}$)')

    plt.ylabel('MRSE [m/s$^2$]')
    plt.xlabel("Parameters")

    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='sci',scilimits=(0, 0),  useMathText=True)

def nn_pareto_curve(file_name, orbit_name, radius_max=None, linestyle=None, marker=None, log=True, sigma=2):
    nn_df = pd.read_pickle(file_name)
    if radius_max is not None:
        sub_df = nn_df[nn_df['radius_max'] == radius_max].sort_values(by='params')
    else:
        sub_df = nn_df
    plt.gca().set_prop_cycle(None)
    if log: 
        plt.semilogx(sub_df['params'], sub_df[orbit_name+'_rse_mean'], linestyle=linestyle, marker=marker)
        plt.semilogx(sub_df['params'], sub_df[orbit_name+'_sigma_'+str(sigma)+'_mean'], linestyle=linestyle, marker=marker)
        plt.semilogx(sub_df['params'], sub_df[orbit_name+'_sigma_'+str(sigma)+'_c_mean'], linestyle=linestyle, marker=marker)
    else:
        plt.plot(sub_df['params'], sub_df[orbit_name+'_rse_mean'], linestyle=linestyle, marker=marker)
        plt.plot(sub_df['params'], sub_df[orbit_name+'_sigma_'+str(sigma)+'_mean'], linestyle=linestyle, marker=marker)
        plt.plot(sub_df['params'], sub_df[orbit_name+'_sigma_'+str(sigma)+'_c_mean'], linestyle=linestyle, marker=marker)        
    plt.legend()
