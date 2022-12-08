from GravNN.Visualization.VisualizationBase import VisualizationBase
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import os
import pandas as pd

class HistoryVisualizer(VisualizationBase):
    def __init__(self, model, config, **kwargs):
        super().__init__(**kwargs)
        self.file_directory += os.path.splitext(os.path.basename(__file__))[0] + "/"
        self.model = model
        self.config = config

    def get_history(self):
        self.history = self.config['history'][0]
        self.epochs = self.history['epochs']
        self.loss = self.history['loss']
        self.val_loss = self.history['val_loss']
        self.percent_error = self.history['percent_error']
        self.val_percent_error = self.history['val_percent_error']

    def plot(self, x, y, label=None, log_x=False, log_y=False):
        if log_x and log_y:
            plt_fcn = plt.loglog
        elif log_x:
            plt_fcn = plt.semilogx
        elif log_y:
            plt_fcn = plt.semilogy
        else:
            plt_fcn = plt.plot
        plt_fcn(x, y, label=label)
        plt.tight_layout()

    def plot_loss(self, **kwargs):
        plt.figure()
        self.plot(self.epochs, self.loss, label='loss', **kwargs)
        self.plot(self.epochs, self.val_loss, label='val loss', **kwargs)