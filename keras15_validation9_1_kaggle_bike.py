
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
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=20, verbose=0, validation_split=0.2)

loss = model.evaluate(x_test, y_test)
print('\nLoss = ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('\nR2 = ', r2)



################  < 작업 결과 >  ##################


[Running] python -u "c:\Users\seongja\OneDrive\바탕 화면\study\keras15_validation9_1_kaggle_bike.py"
2023-03-28 12:33:49.655274: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library ...
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

  1/103 [..............................] - ETA: 4s - loss: 31409.9531
 12/103 [==>...........................] - ETA: 0s - loss: 23302.9688
 32/103 [========>.....................] - ETA: 0s - loss: 24150.1367
 51/103 [=============>................] - ETA: 0s - loss: 24897.5391
 69/103 [===================>..........] - ETA: 0s - loss: 24264.3730
 90/103 [=========================>....] - ETA: 0s - loss: 24227.8359
103/103 [==============================] - 0s 3ms/step - loss: 24520.1445

Loss =  24520.14453125

  1/103 [..............................] - ETA: 38s
 22/103 [=====>........................] - ETA: 0s 
 42/103 [===========>..................] - ETA: 0s
 64/103 [=================>............] - ETA: 0s
 86/103 [========================>.....] - ETA: 0s
103/103 [==============================] - 1s 2ms/step

R2 =  0.2452326770457245

[Done] exited with code=0 in 169.334 seconds

  
