from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from GravNN.Preprocessors.PreprocessorBase import PreprocessorBase
import copy
class MinMaxTransform(PreprocessorBase):
    def __init__(self, val_range=(0,1)):
        super().__init__()
        self.r_scaler = MinMaxScaler(feature_range=val_range)#(0, 1)) # If the data is transformed from r -> 1/r^2 the bounds become [0 (infinity), 1(max)] for body
        self.theta_scaler = MinMaxScaler(feature_range=val_range) # discrete range of values so min max makes sense
        self.phi_scaler = MinMaxScaler(feature_range=val_range) # discrete range of values so min max makes sense

        self.a_r_scaler = MinMaxScaler(feature_range=val_range) # Can't have acceleration in negative radial direction
        self.a_theta_scaler = MinMaxScaler(feature_range=val_range)
        self.a_phi_scaler = MinMaxScaler(feature_range=val_range)

    