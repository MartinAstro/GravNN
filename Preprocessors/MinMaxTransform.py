from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from Preprocessors.PreprocessorBase import PreprocessorBase

class MinMaxTransform(PreprocessorBase):
    r_scaler = MinMaxScaler(feature_range=(0,1))#(0, 1)) # If the data is transformed from r -> 1/r^2 the bounds become [0 (infinity), 1(max)] for body
    theta_scaler = MinMaxScaler(feature_range=(0, 1)) # discrete range of values so min max makes sense
    phi_scaler = MinMaxScaler(feature_range=(0, 1)) # discrete range of values so min max makes sense

    a_r_scaler = MinMaxScaler(feature_range=(0, 1)) # Can't have acceleration in negative radial direction
    a_theta_scaler = MinMaxScaler(feature_range=(0, 1))
    a_phi_scaler = MinMaxScaler(feature_range=(0, 1))

    def apply_transform(self):
        super().apply_transform() #Formats data and checks if data has already been processed

        input_train = self.input_train
        input_test = self.input_test
        output_train = self.output_train
        output_test = self.output_test

        input_train[:, :, 0] = self.r_scaler.fit_transform(input_train[:, :, 0])
        input_train[:, :, 1] = self.theta_scaler.fit_transform(input_train[:, :, 1])
        input_train[:, :, 2] = self.phi_scaler.fit_transform(input_train[:, :, 2])
        output_train[:, :, 0] = self.a_r_scaler.fit_transform(output_train[:, :, 0])
        output_train[:, :, 1] = self.a_theta_scaler.fit_transform(output_train[:, :, 1])
        output_train[:, :, 2] = self.a_phi_scaler.fit_transform(output_train[:, :, 2])

        input_test[:, :, 0] = self.r_scaler.transform(input_test[:, :, 0])
        input_test[:, :, 1] = self.theta_scaler.transform(input_test[:, :, 1])
        input_test[:, :, 2] = self.phi_scaler.transform(input_test[:, :, 2])
        output_test[:, :, 0] = self.a_r_scaler.transform(output_test[:, :, 0])
        output_test[:, :, 1] = self.a_theta_scaler.transform(output_test[:, :, 1])
        output_test[:, :, 2] = self.a_phi_scaler.transform(output_test[:, :, 2])

        self.input_train = input_train.reshape(len(input_train), -1)
        self.input_test = input_test.reshape(len(input_test), -1)

        self.output_train = output_train.reshape(len(output_train), -1)
        self.output_test = output_test.reshape(len(output_test), -1)

        return self.input_train, self.input_test, self.output_train, self.output_test
        
    def invert_transform(self, output_pred=None):
        super().invert_transform()
        input_train = self.input_train.reshape(len(self.input_train), int(len(self.input_train[0]) / 6), 6)
        input_test = self.input_test.reshape(len(self.input_test), int(len(self.input_test[0]) / 6), 6)

        input_train[:, :, 0] = self.r_scaler.inverse_transform(input_train[:, :, 0])
        input_train[:, :, 1] = self.theta_scaler.inverse_transform(input_train[:, :, 1])
        input_train[:, :, 2] = self.phi_scaler.inverse_transform(input_train[:, :, 2])
        input_train[:, :, 3] = self.a_r_scaler.inverse_transform(input_train[:, :, 3])
        input_train[:, :, 4] = self.a_theta_scaler.inverse_transform(input_train[:, :, 4])
        input_train[:, :, 5] = self.a_phi_scaler.inverse_transform(input_train[:, :, 5])

        input_test[:, :, 0] = self.r_scaler.inverse_transform(input_test[:, :, 0])
        input_test[:, :, 1] = self.theta_scaler.inverse_transform(input_test[:, :, 1])
        input_test[:, :, 2] = self.phi_scaler.inverse_transform(input_test[:, :, 2])
        input_test[:, :, 3] = self.a_r_scaler.inverse_transform(input_test[:, :, 3])
        input_test[:, :, 4] = self.a_theta_scaler.inverse_transform(input_test[:, :, 4])
        input_test[:, :, 5] = self.a_phi_scaler.inverse_transform(input_test[:, :, 5])

        self.input_train = input_train.reshape(len(input_train), -1)
        self.input_test = input_test.reshape(len(input_test), -1)

        return output_pred