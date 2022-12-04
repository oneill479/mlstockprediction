#!/usr/bin/env python3
import gi
import warnings
import easygui
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import Adam
from keras.models import Sequential

gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib
# ignore deprecation warning for wmclass
warnings.filterwarnings("ignore", category=DeprecationWarning) 

####################################################################################################
# INITIALIZATION
####################################################################################################

class LSTM:
    def __init__(self):
        # get gui from glade file
        self.builder = Gtk.Builder()
        self.builder.add_from_file("spyML.glade")
        self.builder.connect_signals(self)
        # window
        self.window = self.builder.get_object("window")
        self.window.set_wmclass ("SPY Stock Prediction", "SPY Stock Prediction")
        self.window.set_title("Machine Learning Stock Prediction")
        self.window.set_position(Gtk.WindowPosition.CENTER_ALWAYS)
        self.window.show_all()
        # entry boxes
        self.traingingDataEntry = self.builder.get_object("traingingDataEntry")
        self.testDataEntry = self.builder.get_object("testDataEntry")
        self.predictCheckBox = self.builder.get_object("predictCheckBox")
        # buttons
        self.runAlgorithmButton = self.builder.get_object("runAlgorithmButton")
        self.defaultChartButton = self.builder.get_object("defaultChartButton")
        self.modelChartButton = self.builder.get_object("modelChartButton")
        self.futurePredictionsButton = self.builder.get_object("futurePredictionsButton")
        # global variables
        self.train = None
        self.validation = None
        self.prediction = None
        self.predictNextDay = True
        self.df = None
        

####################################################################################################
# FUNTIONS
####################################################################################################

    def start_algorithm(self, widget, data=None):

        # pop up alert for training entry box not being filled
        if not self.traingingDataEntry.get_text():
            easygui.msgbox("You must fill out all training data box!", title="Error")
            return

        # set training data percentage
        trainingData = self.traingingDataEntry.get_text().strip('%')
        self.testDataEntry.set_text(str(100 - int(trainingData)) + '%')
        trainingData = int(trainingData)
        trainingData = trainingData / 100

        # set buttons to false when running algorithm
        self.defaultChartButton.set_sensitive(False)
        self.modelChartButton.set_sensitive(False)
        self.futurePredictionsButton.set_sensitive(False)

        # Get the Dataset
        self.df=pd.read_csv("spy.csv", na_values=['null'], index_col='Date', parse_dates=True, infer_datetime_format=True)
        self.df.head()
        
        # get closing price of each day
        last_prices = self.df['Last']
        values = last_prices.values
        train_data_length = math.ceil(len(values) * trainingData)

        # scale down data to values between 0 and 1
        scaler = MinMaxScaler(feature_range=(0,1))
        feature_transform = scaler.fit_transform(values.reshape(-1,1))
        train_data = feature_transform[0: train_data_length, :]

        # training data lists
        x_train_data = []
        y_train_data = []

        # create a window of 90 days of data
        for d in range(90, len(train_data)):
            x_train_data.append(train_data[d-90:d, 0])
            y_train_data.append(train_data[d, 0])

        # reshape data into a 3-dimensional array for LSTM
        x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

        test_data_set = feature_transform[train_data_length-90: , : ]
        test_data = []
        y_test = values[train_data_length:]

        for d in range(90, len(test_data_set)):
            test_data.append(test_data_set[d-90:d, 0])

        test_data = np.array(test_data)
        test_data = np.reshape(test_data, (test_data.shape[0], test_data.shape[1], 1))

        # set up LSTM network architecture
        model = Sequential()
        model.add(keras.layers.LSTM(128, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        model.add(keras.layers.LSTM(64, return_sequences=False))
        model.add(keras.layers.Dense(25))
        model.add(keras.layers.Dense(1))
        model.summary()

        # set optimizer and loss functions
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train_data, y_train_data, batch_size= 1, epochs=5)

        predictions = model.predict(test_data)
        predictions = scaler.inverse_transform(predictions)
        rmse = np.sqrt(np.mean(predictions - y_test)**2)
        print()
        print('RMSE', rmse)
        print()

        # set the prediction values
        predictions = model.predict(test_data)
        predictions = scaler.inverse_transform(predictions)

        # seperate data for pyplot
        data = self.df.filter(['Last'])
        self.train = data[:train_data_length]
        self.validation = data[train_data_length:]
        self.validation['Prediction'] = predictions

        # predict the next day stock price
        real_data = [test_data_set[len(test_data_set)+1 - len(data[train_data_length:]):len(test_data_set+1),0]]
        real_data = np.array(real_data)
        real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
        self.prediction = model.predict(real_data)
        self.prediction = scaler.inverse_transform(self.prediction)

        # set buttons to true once training is complete
        self.defaultChartButton.set_sensitive(True)
        self.modelChartButton.set_sensitive(True)
        if self.predictNextDay: self.futurePredictionsButton.set_sensitive(True)  

    # show the default stock chart
    def default_chart(self, widget, data=None):
        plt.figure(figsize=(15, 8))
        plt.title('SPY Price History')
        plt.plot(self.df['Last'])
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.show()
    # show the machine learning model chart
    def model_chart(self, widget, data=None):
        plt.figure(figsize=(16,8))
        plt.title('SPY Prediction Model')
        plt.xlabel('Date')
        plt.ylabel('Close ($)')
        plt.plot(self.train)
        plt.plot(self.validation[['Last', 'Prediction']])
        plt.legend(['Train', 'Val', 'Prediction'], loc='lower right')
        plt.show()

    # future prediction scatter plot
    def future_predictions(self, widget, data=None):
        decision = ''
        scatterArr = list(self.df['Last'][-10:-1])
        dateArr = [i for i in range(1, 10)]
        plt.scatter(dateArr, scatterArr)
        # decide whether to buy or sell
        if self.prediction > scatterArr[-1]:
            decision = "green"
        else:
            decision = "red"
        # next day prediction dot
        plt.scatter([10], [float(self.prediction)], c=decision)
        # create legend
        green = mpatches.Patch(color='green', label='Buy')
        red = mpatches.Patch(color='red', label='Sell')
        blue = mpatches.Patch(color='blue', label='Previous 9 Days')
        scatterArr.append(float(self.prediction))
        m, b = np.polyfit(np.array([1,2,3,4,5,6,7,8,9,10]), scatterArr, 1)
        plt.plot(np.array([1,2,3,4,5,6,7,8,9,10]), m * np.array([1,2,3,4,5,6,7,8,9,10]) + b, color='orange')
        plt.legend(handles=[green, red, blue])
        plt.annotate(str(round(float(self.prediction), 2)), xy=(10, float(self.prediction)), xytext=(10.2, float(self.prediction)))
        plt.show()
        
    # check box to get a next day prediction
    def predict_checkbox(self, widget, data=None):
        if self.predictCheckBox.get_active():
            self.predictNextDay = True
        else:
            self.predictNextDay = False

    # dynamically update test data box
    def change_test_data(self, widget, data=None):
        if self.traingingDataEntry.get_text() != '':
            trainingData = self.traingingDataEntry.get_text().strip('%')
            self.testDataEntry.set_text(str(100 - int(trainingData)) + '%')
        
    # when window is closed shut down process
    def on_window_destroy(self, widget, data=None):
        print ("Window destroyed!")
        Gtk.main_quit()
    # main function launching Gtk application
    def main(self):
            Gtk.main()

# main
if __name__ == "__main__":
    application = LSTM()
    application.main()
