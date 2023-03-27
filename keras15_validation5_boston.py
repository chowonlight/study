
import numpy as np
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets.target

print('\n', x.shape, y.shape, '\n')     

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=29)

model = Sequential()
model.add(Dense(64, input_dim=13, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=10, 
            validation_split=0.2, verbose=0)

loss = model.evaluate(x_test, y_test)
print('\nLoss = ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('\nR2 = ', r2)

################  < 작업 결과 >  ##################


 (506, 13) (506,) 

2023-03-27 20:41:54.816955: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library ...
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

1/4 [======>.......................] - ETA: 0s - loss: 19.5114
4/4 [==============================] - 0s 14ms/step - loss: 15.8865

Loss =  15.886496543884277

R2 =  0.7909379034632477

[Done] exited with code=0 in 203.766 seconds
