# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 16:21:15 2018

@author: hugob
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras



data = pd.read_csv(filepath_or_buffer="train_1.csv")
dataset = data.values
dataset.shape
pages = dataset[:,0]
X = dataset[:1000,1:]
time = data.columns[1:]

plt.plot(time, X[0,:])
plt.legend()
plt.show()

for i in range(len(X)-1,-1,-1):
    j = 0
    while j<len(X[0,]):
        if(math.isnan(X[i,j])):
            X = np.delete(X, (i), axis = 0)
            j = len(X[0,])
        j += 1


import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


#plt.plot(X)
#plt.ylabel(" UK pension")
#plt.xlabel("time")
#plt.show()

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)][0]
        dataX.append(a)
        dataY.append(dataset[(i+look_back)])
    return np.array(dataX), np.array(dataY)

# prediction for the selected time serie
def modele(rang):
    # fix random seed for reproducibility
    np.random.seed(7)
    
    serie = np.zeros((len(X[rang]),1))
    for k in range(len(X[rang])):
        serie[k] = X[rang,k]
    
    serie.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    serie = scaler.fit_transform(serie)
    
    train_size = int(len(serie) * 0.67)
    test_size = len(serie) - train_size
    
    # split into train and test sets
    train = serie[0:train_size]
    test = serie[train_size:len(serie)]
    
    # reshape into X=t and Y=t+1
    look_back = 1
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    
    
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
    
    
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    
    
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    testPredict = scaler.inverse_transform(testPredict)
    
    trainY = np.reshape(trainY, (trainY.shape[0]))
    testY = np.reshape(testX, (testX.shape[0]))
    trainY = scaler.inverse_transform([trainY])
    testY = scaler.inverse_transform([testY])
    
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    
    # shift train predictions for plotting
    trainPredictPlot = np.empty_like(serie)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = np.empty_like(serie)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(serie)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(serie))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.xlabel('Time')
    plt.ylabel('Time serie')
    plt.legend()
    plt.show()
    
    
#    # fit network
#    history = model.fit(trainX, trainY, epochs=20, batch_size=64, validation_data=(testX, testY), verbose=2, shuffle=False)
#    
#    # summarize history for loss
#    plt.plot(history.history['loss'])
#    plt.plot(history.history['val_loss'])
#    plt.title('model loss')
#    plt.ylabel('loss')
#    plt.xlabel('epoch')
#    plt.legend(['train', 'test'], loc='upper right')
#    plt.show()
    
def score(nb):
    trainScoreTot = []
    testScoreTot = []
    
    for rang in range(nb):
        # fix random seed for reproducibility
        np.random.seed(7)
        
        serie = np.zeros((len(X[rang]),1))
        for k in range(len(X[rang])):
            serie[k] = X[rang,k]
        
        serie.astype('float32')
        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        serie = scaler.fit_transform(serie)
        
        train_size = int(len(serie) * 0.67)
        test_size = len(serie) - train_size
        
        # split into train and test sets
        train = serie[0:train_size]
        test = serie[train_size:len(serie)]
        
        # reshape into X=t and Y=t+1
        look_back = 1
        trainX, trainY = create_dataset(train, look_back)
        testX, testY = create_dataset(test, look_back)
        
        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        
        
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(4, input_shape=(1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='sgd')
        model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)
        
        # make predictions
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        
        
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        testPredict = scaler.inverse_transform(testPredict)
        
        trainY = np.reshape(trainY, (trainY.shape[0]))
        testY = np.reshape(testX, (testX.shape[0]))
        trainY = scaler.inverse_transform([trainY])
        testY = scaler.inverse_transform([testY])
        
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))
        
        trainScoreTot.append(trainScore)
        testScoreTot.append(testScore)
        
    trainScoreMean = np.mean(trainScoreTot)
    print('Train Score mean: %.2f RMSE' % (trainScoreMean))
    testScoreMean = np.mean(testScoreTot)
    print('Test Score mean: %.2f RMSE' % (testScoreMean))
    
    trainScoreMax = np.max(trainScoreTot)
    print('Train Score max: %.2f RMSE' % (trainScoreMax))
    trainScoreMaxArg = np.argmax(trainScoreTot)
    print(trainScoreMaxArg)
    testScoreMax = np.max(testScoreTot)
    print('Test Score max: %.2f RMSE' % (testScoreMax))
    testScoreMaxArg = np.argmax(testScoreTot)
    print(testScoreMaxArg)
    
    trainScoreMin = np.min(trainScoreTot)
    print('Train Score min: %.2f RMSE' % (trainScoreMin))
    trainScoreMinArg = np.argmin(trainScoreTot)
    print(trainScoreMinArg)
    testScoreMin = np.min(testScoreTot)
    print('Test Score min: %.2f RMSE' % (testScoreMin))
    testScoreMinArg = np.argmin(testScoreTot)
    print(testScoreMinArg)
    
    