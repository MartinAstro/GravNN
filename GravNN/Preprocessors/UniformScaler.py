import numpy as np


class UniformScaler:
    def __init__(self, feature_range=(-1, 1)):
        """Scale the variable by the min and max of the range or by a constant scaler"""
        self.feature_range = feature_range
        pass

    def fit(self, data, scaler=None):
        self.scaler = scaler

        data_max = np.max(data)
        data_min = np.min(data)

        data_range = data_max - data_min
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range

    def fit_transform(self, data, scaler=None):
        self.scaler = scaler
        data_max = np.max(data)
        data_min = np.min(data)

        data_range = data_max - data_min
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / data_range
        self.min_ = self.feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range

        if self.scaler is not None:
            X = data * self.scaler
            self.scale_ = self.scaler
            self.min_ = 0.0
        else:
            X = data * self.scale_ + self.min_
        return X

    def transform(self, data):
        # Some older networks will load an old version of uniform scaler
        # that didn't have scaler attribute. Therefore, add it if not present
        if not hasattr(self, 'scaler'):
            self.scaler = None

        if self.scaler is not None:
            X = data * self.scaler
        else:
            X = data * self.scale_ + self.min_
        return X

    def inverse_transform(self, data):
        # Some older networks will load an old version of uniform scaler
        # that didn't have scaler attribute. Therefore, add it if not present
        if not hasattr(self, 'scaler'):
            self.scaler = None
        if self.scaler is not None:
            return data / self.scaler
        else:
            return (data - self.min_) / self.scale_
