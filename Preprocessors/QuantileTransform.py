from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from GravNN.Preprocessors.PreprocessorBase import PreprocessorBase
import copy
class QuantileTransform(PreprocessorBase):
    def __init__(self):
        super().__init__()
        self.r_scaler = QuantileTransformer(output_distribution='normal', random_state=0)#(0, 1)) # If the data is transformed from r -> 1/r^2 the bounds become [0 (infinity), 1(max)] for body
        self.theta_scaler = QuantileTransformer(output_distribution='normal', random_state=0) # discrete range of values so min max makes sense
        self.phi_scaler = QuantileTransformer(output_distribution='normal', random_state=0) # discrete range of values so min max makes sense

        self.a_r_scaler = QuantileTransformer(output_distribution='normal', random_state=0) # Can't have acceleration in negative radial direction
        self.a_theta_scaler = QuantileTransformer(output_distribution='normal', random_state=0)
        self.a_phi_scaler = QuantileTransformer(output_distribution='normal', random_state=0)
