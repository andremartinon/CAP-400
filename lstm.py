import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


class TimeSeriesLSTM:

    def __init__(self, dataset: np.ndarray, look_back: int = 1):
        dataset = dataset.reshape(-1, 1)

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        # split into train and test sets
        train_size = int(len(dataset) * 0.67)
        train = dataset[0:train_size, :]
        test = dataset[train_size:len(dataset), :]

        # reshape into X=t and Y=t+1
        train_x, train_y = TimeSeriesLSTM._create_dataset(train, look_back)
        test_x, test_y = TimeSeriesLSTM._create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

        self.look_back = look_back
        self.dataset = dataset
        self.scaler = scaler

        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

        self.train_score = None
        self.test_score = None

        self.train_predict = None
        self.test_predict = None

        self.epochs = None
        self.batch_size = None

    # convert an array of values into a dataset matrix
    @staticmethod
    def _create_dataset(dataset, look_back=1):
        data_x, data_y = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            data_x.append(a)
            data_y.append(dataset[i + look_back, 0])
        return np.array(data_x), np.array(data_y)

    def train(self, epochs: int = 50, batch_size: int = 1, verbose: int = 0):
        self.epochs = epochs
        self.batch_size = batch_size

        # fix random seed for reproducibility
        np.random.seed(7)

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, self.look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(self.train_x,
                  self.train_y,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=verbose)

        # make predictions
        train_predict = model.predict(self.train_x)
        test_predict = model.predict(self.test_x)

        # invert predictions
        self.train_predict = self.scaler.inverse_transform(train_predict)
        train_y = self.scaler.inverse_transform([self.train_y])
        self.test_predict = self.scaler.inverse_transform(test_predict)
        test_y = self.scaler.inverse_transform([self.test_y])

        # calculate root mean squared error
        self.train_score = math.sqrt(
            mean_squared_error(train_y[0], self.train_predict[:, 0]))
        self.test_score = math.sqrt(
            mean_squared_error(test_y[0], self.test_predict[:, 0]))

    def plot(self, show: bool = True, title: str = '',
             x_label: str = 'Snapshots', y_label: str = 'Variable',
             file_name: Path = None):

        # shift train predictions for plotting
        train_predict_plot = np.empty_like(self.dataset)
        train_predict_plot[:, :] = np.nan
        train_predict_plot[self.look_back:len(self.train_predict) + self.look_back, :] = self.train_predict

        # shift test predictions for plotting
        test_predict_plot = np.empty_like(self.dataset)
        test_predict_plot[:, :] = np.nan
        test_predict_plot[len(self.train_predict) + (self.look_back * 2) + 1:len(self.dataset) - 1, :] = self.test_predict

        # plot baseline and predictions
        plt.figure(figsize=(11.7, 3.3))
        plt.plot(self.scaler.inverse_transform(self.dataset))
        plt.plot(train_predict_plot)
        plt.plot(test_predict_plot)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        plt.subplots_adjust(top=0.9, bottom=0.15, left=0.08, right=0.98,
                            hspace=0.15, wspace=0.1)

        plt.legend(handlelength=0, handletextpad=0,
                   labels=[f'Epochs: {self.epochs}, Batch: {self.batch_size}',
                           f'Train Score: {self.train_score:.6f} RMSE',
                           f'Test Score: {self.test_score: .6f} RMSE'],
                   ncol=1, fontsize=10)

        if show:
            plt.show()
        else:
            plt.savefig(file_name, format='png', dpi=300)
        plt.close()
