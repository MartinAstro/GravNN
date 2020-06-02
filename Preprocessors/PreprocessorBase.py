from abc import ABC, abstractmethod
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
class PreprocessorBase:
    input_train = None
    input_test = None
    output_train = None
    output_test = None
    percentTest = 0.1

    preprocessed = False
    def __init__(self):
        return
        
    def split(self, position, acceleration):
        self.input_train, self.input_test, self.output_train, self.output_test = train_test_split(position, acceleration, test_size=self.percentTest, random_state=13)

        return   self.input_train, self.input_test, self.output_train, self.output_test
    

    def apply_transform(self):
        if self.input_train is None or self.output_train is None:
            exit("Data Error: Need to specify training data before pre-processing!")
        if self.preprocessed:
            exit("Data Error: The data is already pre-processed!")

        self.input_train = self.input_train.reshape(len(self.input_train), int(len(self.input_train[0]) / 3), 3)
        self.input_test = self.input_test.reshape(len(self.input_test), int(len(self.input_test[0]) / 3), 3)

        self.output_train = self.output_train.reshape(len(self.output_train), int(len(self.output_train[0]) / 3), 3)
        self.output_test = self.output_test.reshape(len(self.output_test), int(len(self.output_test[0]) / 3), 3)

        self.preprocessed = True
        return 

    def invert_transform(self):
        '''
        Return data to original scaling
        If a prediction was provided, also scale that to original values
        :param output_pred: Predicted output values to scale
        :return: Scaled predicted output
        '''
        if not self.preprocessed:
            exit("Data Error: The data was never preprocessed, and therefore can't be post-processed!")

        self.preprocessed = False
        return 