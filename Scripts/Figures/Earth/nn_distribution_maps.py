
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import sigfig
import os 
import subprocess

file_name = 'C:\\Users\\John\\Documents\\Research\\ML_Gravity\\N_1000000_exp_norm_study.data'

df = pd.read_pickle(file_name).sort_values(by='params', ascending=False).iloc[0:2]
print(df[['distribution', 'params', 'invert', 'mu', 'sigma', 'scale_parameter']])

for index, row in df.iterrows():
    index = row.name
    print(os.path.abspath('.') + "/Data/Networks/")
    plot_directory = os.path.abspath('.') + "/Data/Networks/" + str(pd.Timestamp(index).to_julian_date()) + "/LEO/pred.pdf"
    subprocess.Popen([plot_directory],shell=True)
    
    plot_directory = os.path.abspath('.') + "/Data/Networks/" + str(pd.Timestamp(index).to_julian_date()) + "/Brillouin/pred.pdf"
    subprocess.Popen([plot_directory],shell=True)

    print(os.path.abspath('.') + "/Data/Networks/")
    plot_directory = os.path.abspath('.') + "/Data/Networks/" + str(pd.Timestamp(index).to_julian_date()) + "/LEO/true.pdf"
    subprocess.Popen([plot_directory],shell=True)
    
    plot_directory = os.path.abspath('.') + "/Data/Networks/" + str(pd.Timestamp(index).to_julian_date()) + "/Brillouin/true.pdf"
    subprocess.Popen([plot_directory],shell=True)