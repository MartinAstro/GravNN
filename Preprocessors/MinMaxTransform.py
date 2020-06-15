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

    def fit(self):
        #Formats data and checks if data has already been processed
        self.input_train = super().format_data(self.input_train) #Formats data and checks if data has already been processed
        self.input_test = super().format_data(self.input_test)
        self.output_train = super().format_data(self.output_train)
        self.output_test = super().format_data(self.output_test)

        self.r_scaler.fit(self.input_train[:, :, 0])
        self.theta_scaler.fit(self.input_train[:, :, 1])
        self.phi_scaler.fit(self.input_train[:, :, 2])
        self.a_r_scaler.fit(self.output_train[:, :, 0])
        self.a_theta_scaler.fit(self.output_train[:, :, 1])
        self.a_phi_scaler.fit(self.output_train[:, :, 2])

    def apply_transform(self, x=None, y=None):
        if x is None and y is None:
            super().apply_transform()
            input_train = self.input_train
            input_test = self.input_test
            output_train = self.output_train
            output_test = self.output_test

            input_train[:, :, 0] = self.r_scaler.transform(input_train[:, :, 0])
            input_train[:, :, 1] = self.theta_scaler.transform(input_train[:, :, 1])
            input_train[:, :, 2] = self.phi_scaler.transform(input_train[:, :, 2])
            output_train[:, :, 0] = self.a_r_scaler.transform(output_train[:, :, 0])
            output_train[:, :, 1] = self.a_theta_scaler.transform(output_train[:, :, 1])
            output_train[:, :, 2] = self.a_phi_scaler.transform(output_train[:, :, 2])

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

        else:
            if x is not None:
                x = super().format_data(x)
                #If the radius of the data is different at the level of machine precision, round the values to be the same. 
                if abs(x[:,:,0].max() - x[:,:,0].min()) < 1E-8:
                    x[:,:,0] = np.round(x[:,:,0], 4)
                x[:, :, 0] = self.r_scaler.transform(x[:, :, 0])
                x[:, :, 1] = self.theta_scaler.transform(x[:, :, 1])
                x[:, :, 2] = self.phi_scaler.transform(x[:, :, 2])
            
            if y is not None:
                y = super().format_data(y)
                y[:, :, 0] = self.a_r_scaler.transform(y[:, :, 0])
                y[:, :, 1] = self.a_theta_scaler.transform(y[:, :, 1])
                y[:, :, 2] = self.a_phi_scaler.transform(y[:, :, 2])

            return x, y
            

        
    def invert_transform(self, x=None, y=None):
        if x is None and y is None:
            super().invert_transform()
            input_train = self.input_train
            input_test = self.input_test
            output_train = self.output_train
            output_test = self.output_test

            input_train[:, :, 0] = self.r_scaler.inverse_transform(input_train[:, :, 0])
            input_train[:, :, 1] = self.theta_scaler.inverse_transform(input_train[:, :, 1])
            input_train[:, :, 2] = self.phi_scaler.inverse_transform(input_train[:, :, 2])
            output_train[:, :, 0] = self.a_r_scaler.inverse_transform(output_train[:, :, 0])
            output_train[:, :, 1] = self.a_theta_scaler.inverse_transform(output_train[:, :, 1])
            output_train[:, :, 2] = self.a_phi_scaler.inverse_transform(output_train[:, :, 2])

            input_test[:, :, 0] = self.r_scaler.inverse_transform(input_test[:, :, 0])
            input_test[:, :, 1] = self.theta_scaler.inverse_transform(input_test[:, :, 1])
            input_test[:, :, 2] = self.phi_scaler.inverse_transform(input_test[:, :, 2])
            output_test[:, :, 0] = self.a_r_scaler.inverse_transform(output_test[:, :, 0])
            output_test[:, :, 1] = self.a_theta_scaler.inverse_transform(output_test[:, :, 1])
            output_test[:, :, 2] = self.a_phi_scaler.inverse_transform(output_test[:, :, 2])

            self.input_train = input_train.reshape(len(input_train), -1)
            self.input_test = input_test.reshape(len(input_test), -1)

            self.output_train = output_train.reshape(len(output_train), -1)
            self.output_test = output_test.reshape(len(output_test), -1)

            return self.input_train, self.input_test, self.output_train, self.output_test

        else:
            if x is not None:
                x[:, :, 0] = self.r_scaler.inverse_transform(x[:, :, 0])
                x[:, :, 1] = self.theta_scaler.inverse_transform(x[:, :, 1])
                x[:, :, 2] = self.phi_scaler.inverse_transform(x[:, :, 2])
                x = super().unformat_data(x)
            if y is not None:
                y[:, :, 0] = self.a_r_scaler.inverse_transform(y[:, :, 0])
                y[:, :, 1] = self.a_theta_scaler.inverse_transform(y[:, :, 1])
                y[:, :, 2] = self.a_phi_scaler.inverse_transform(y[:, :, 2])
                y = super().unformat_data(y)

            
            return x, y