# Goal is to build a RNN model oto predict open Google stock price on completely new test data
# Part 1: Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Import training set- we train the model only on training set


## predicitions are made on completely new test set
dataset_train = pd.read_csv(r"C:\Users\Administrator\.spyder-py3\RNN_dataset\Google_Stock_Price_Train.csv")

## we only need open stock price values
training_set = np.array(dataset_train.iloc[:,1:2])

## feature scaling, using normalisation, viz x-xmax/(xmax-xmin)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1)) # default arguments, this says that all the scaled values are between 0 and 1
training_set_scaled = sc.fit_transform(training_set) # here, fit implies that we only get min of the training data, transform does the actual transformation

## very important step of preprocessing for RNNs especially
## creating a datastructure with n number of timesteps
## let's set 60 timesteps, i.e, at each time t, it will look at 60 stock prices before time t, then it will capture the corresponding trends
## this correspnds to 3 months(only include financial days)
## for each X_train, it would contain th 60 prev stock prices & Y_train contains the stock price on the next day
X_train = []
Y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0]) # we need all prices from i-60 to i
    Y_train.append(training_set_scaled[i, 0])
    
X_train = np.array(X_train)
Y_train = np.array(Y_train)

number_of_indicators = 1
## this will be used in the last column, so that we can add extra info - this could be google's closing stock price or any other company's opening stock price, to help the model learn better
## last step- reshaping, to add a new dimension corresponding to the unit- the number of predictors
## it is done to add another indicator of upward and downward trends
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], number_of_indicators)) # converting to 3D- chk keras documentation- in the paranthesis, we shd have batch_size(total number of obs), then timesteps, then input_dim


#2. Building the RNN Architecture using the LSTM
## a stakced LSTM type of architecture with high diemsnionality

## some pacakges
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout     # include dropout regularization

## initialising the RNN
regressor = Sequential()    # we will be predicting continuous stock prices, therefore not a classifier in CNN

## adding the first LSTM layer, then add dropout normalisation to prevent overfitiing
regressor.add(LSTM(units = 50, return_sequences=True, input_shape = (X_train.shape[1], number_of_indicators)))   # in LSTM, we need to provide number of LSTM cells in the 1st layer, return sequences(default is false), input shape, i.e,last 2 dimensions from the 3D X_train
## because we are adding other LSTM layers, we need to set return sequences to True

## adding the dropout regularisation
regressor.add(Dropout(rate = 0.2))    # dropout rate- rate at which we ignore neurons in the training involving forward and back prop in every iterations, here, 10 neurons are dropped out in every training

## adding more LSTM layers
regressor.add(LSTM(units = 50, return_sequences=True))  # we don't need to add extra input shape after 1st layer, the units argument would automaticaly detect shape
regressor.add(Dropout(rate = 0.2))

regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(rate = 0.2))

regressor.add(LSTM(units = 50, return_sequences = False))   # this is the last LSTM layer
regressor.add(Dropout(rate = 0.2))

## adding the output layer
regressor.add(Dense(units = 1))

## Compiling
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
## rmsprop is an advanced stochastic gradient optimizer, go to choice for RNN, but we will actually use Adam, as it gives better results

## Fitting
regressor.fit(X_train, Y_train, epochs = 100, batch_size = 32)  # epochs is the number of times for the entire forward and back prop, 32 of them are sent forward and backword in 1 epcoh

#3. Making predictions and visualisation

## first we get the actual test set
dataset_test = pd.read_csv(r"C:\Users\Administrator\.spyder-py3\RNN_dataset\Google_Stock_Price_Test.csv")
real_stockPrices = np.array(dataset_test.iloc[:,1:2])

## getting the predicted stock price for 2017 data
## to predict too, we need 60 prev days' data, this is done by concatenating training and testing set values

## never change actual test values, i.e, don't scale them, but we need some train data for the concatenation, therefore we will use the very first raw dataset_train 
dataset_whole = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)  # concatenate the rows
## this is the new input of 60 data points
inputs = np.array(dataset_whole[len(dataset_whole)-len(dataset_test)-60: ])    # the start number represents first financial day of '17
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)   # don't fit here, we need to use the same mean and std

## let's reshape it to a proper special 3D structure so that it fits the prediction part too
X_test = []
### we are only predictions, no need of Y_test
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0]) # we need all prices from i-60 to i


X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], number_of_indicators))

#4. Prediction
predicted_stock_price = regressor.predict(X_test)
## the values are predicted_stock_price are already scaled, we need to invert this
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#5. Some visualisations
## Real Stock Prices
plt.plot(real_stockPrices, color = 'blue', label = 'True Google Stock Prices', marker = 'o')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted Google Stock Prices')

plt.title('Google Stock Prices: Real vs Predicted')
plt.xlabel('Days')
plt.ylabel('Value')
plt.xticks(np.arange(0,21))
plt.legend()
plt.show()

