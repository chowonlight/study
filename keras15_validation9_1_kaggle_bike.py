
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

train_csv = train_csv.dropna()   

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=123)

model = Sequential()
model.add(Dense(32, input_dim=8))
model.add(Dense(64))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=36, verbose=0, validation_split=0.2)

loss = model.evaluate(x_test, y_test)
print('\nLoss = ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('\nR2 = ', r2)



################  < 작업 결과 >  ##################


[Running] python -u "c:\Users\seongja\OneDrive\바탕 화면\study\keras15_validation9_2_kaggle_bike"
2023-03-28 13:08:33.852102: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library ...
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

  1/103 [..............................] - ETA: 4s - loss: 31417.9258
 10/103 [=>............................] - ETA: 0s - loss: 23715.9570
 29/103 [=======>......................] - ETA: 0s - loss: 23603.6602
 51/103 [=============>................] - ETA: 0s - loss: 24821.0078
 72/103 [===================>..........] - ETA: 0s - loss: 24138.8984
 89/103 [========================>.....] - ETA: 0s - loss: 24069.9609
103/103 [==============================] - 0s 3ms/step - loss: 24458.9180

Loss =  24458.91796875

  1/103 [..............................] - ETA: 30s
 27/103 [======>.......................] - ETA: 0s 
 55/103 [===============>..............] - ETA: 0s
 96/103 [==========================>...] - ETA: 0s
103/103 [==============================] - 0s 2ms/step

R2 =  0.24711725291789421

[Done] exited with code=0 in 218.063 seconds

  
