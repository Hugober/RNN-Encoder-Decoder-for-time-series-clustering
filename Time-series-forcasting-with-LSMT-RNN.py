import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras

#tensorflow importation test
#hello = tf.constant('Hello, TensorFlow!')
#sess = tf.Session()
#print(sess.run(hello))

data = pd.read_csv(filepath_or_buffer="UK_pension.csv",encoding='latin1',usecols=[1], engine='python', skipfooter=3)
dataset = data.values
dataset.shape
#plt.plot(data.value)
#plt.legend()
#plt.show()

import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# fix random seed for reproducibility
np.random.seed(7)

pension = dataset
pension = pension.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
pension = scaler.fit_transform(pension)

plt.plot(pension)
plt.ylabel(" UK pension")
plt.xlabel("time")
plt.show()

# split into train and test sets
train_size = int(len(pension) * 0.67)
test_size = len(pension) - train_size
train, test = pension[0:train_size], pension[train_size:len(pension)]
print(len(train), len(test))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)][0]
        dataX.append(a)
        dataY.append(dataset[(i+look_back)])
    return np.array(dataX), np.array(dataY)

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
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


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

dataset=pension
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.xlabel('Time')
plt.ylabel('UK quaterly pensions')
plt.legend()
plt.show()


# fit network
history = model.fit(trainX, trainY, epochs=20, batch_size=64, validation_data=(testX, testY), verbose=2, shuffle=False)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()