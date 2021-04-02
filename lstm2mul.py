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

cdf = scaler.inverse_transform(cdf)
plt.plot(cdf, 'g', label='30MAD12CY001 XQ01')
plt.plot(trainPredictPlot, 'r', label='Training Set')
plt.plot(testPredictPlot, 'b', label='Predicted value')
plt.title("LSTM Predicted")
plt.xlabel('Time', fontsize=12)
plt.ylabel('Bearing vibration', fontsize=12)
plt.legend(loc='upper right')
plt.show()
