import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
xsize=12
yoffset=36
features = 4
L=max(xsize, yoffset)
# Crate 1D Data into Time-Series
def new_dataset(dataset,ycol):
	data_X, data_Y = [], []
	for i in range(len(dataset)-L):
		data_X.append(dataset[i:(i+xsize), :])
		data_Y.append(dataset[i + yoffset, ycol])
	return np.array(data_X), np.array(data_Y)

#Load Dataset
df = pd.read_csv('/home/asm/m/k/lstmJoin.csv')
# We convert Date column to datetime
df['datetime']=pd.to_datetime(df['date'])
df.index=df['datetime']
df=df.drop(['date'],axis=1)

df = df.resample('60T').mean()

#<, extend to multi dimentional input> 
# Reindex all of dataset by Date column
#df = df.reindex(index= df.index[::-1])

#zaman = np.arange(1, len(df) + 1, 1)
cdf = df#.mean(axis=1)
#plt.plot(cdf)
#plt.show()
#print(cdf.head())

# Normalize dataset
columns =df.columns.drop('datetime')
df[columns] = scaler.fit_transform(df[columns])
#print(cdf)


# Normalize dataset
cdf = scaler.fit_transform(cdf)
cdf = np.reshape(cdf.values, (len(cdf),1)) #7288 data
scaler = MinMaxScaler(feature_range=(0,1))
#Train-Test SPLIT dataset
split = int(len(cdf)*0.56)
train_d, test_d = cdf[0:split,:], cdf[split:len(cdf),:]
# We create 1D dimension dataset from mean OHLV
trainX, trainY = new_dataset(train_d)
testX, testY = new_dataset(test_d)


# Reshape dataset for LSTM in 3D Dimension
trainX = np.reshape(trainX, (trainX.shape[0],xsize,features))
testX = np.reshape(testX, (testX.shape[0],xsize,features))

# LSTM Model is created
model = Sequential()
model.add(LSTM(64, input_shape=(xsize,features), return_sequences=True))
#model.add(Dropout(0.1))
model.add(LSTM(64, return_sequences=True))
#model.add(Dropout(0.1))
model.add(LSTM(64))
#model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=10, batch_size=25, verbose=2)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DE-Normalizing for plotting 
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Performance Measure RMSE is calculated for predicted train dataset
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print("Train RMSE: %.2f" % (trainScore))

# Performance Measure RMSE is calculated for predicted test dataset
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print("Test RMSE: %.2f" % (testScore))

# Converted predicted train dataset for plotting
trainPredictPlot = np.empty_like(cdf)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[yoffset:len(trainPredict)+yoffset,:] = trainPredict

# Converted predicted test dataset for plotting
testPredictPlot = np.empty_like(cdf)
testPredictPlot[:,:] = np.nan
testPredictPlot[split+yoffset:split+yoffset+len(testPredict),:] = testPredict


# Finally predicted values are visualized
cdf = scaler.inverse_transform(cdf)
plt.plot(cdf, 'g', label='30MAD12CY001 XQ01')
plt.plot(trainPredictPlot, 'r', label='Training Set')
plt.plot(testPredictPlot, 'b', label='Predicted value')
plt.title("LSTM Predicted")
plt.xlabel('Time', fontsize=12)
plt.ylabel('Bearing vibration', fontsize=12)
plt.legend(loc='upper right')
plt.show()
