
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

train_csv = train_csv.dropna()

x = train_csv.drop(['count'], axis=1)
y = train_csv['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123)


model = Sequential()
model.add(Dense(32, input_dim=9))
model.add(Dense(64))
model.add(Dense(124))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=300, batch_size=15, validation_split=0.2, verbose=0)


loss = model.evaluate(x_test, y_test)
print('\nLoss = ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('\nR2 = ', r2)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

print('\nRMSE = ', rmse)


submission = pd.read_csv(path + 'submission.csv', index_col=0)
y_submit = model.predict(test_csv)
submission['count'] = y_submit

submission.to_csv(path_save + 'submit_0328_0101.csv')



################  < 작업 결과 >  ##################


[Running] python -u "c:\Users\seongja\OneDrive\바탕 화면\study\keras15_validation8_1_ddarung.py"
2023-03-28 11:43:10.270958: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library ...
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

 1/13 [=>............................] - ETA: 0s - loss: 1136.4735
13/13 [==============================] - ETA: 0s - loss: 2825.5269
13/13 [==============================] - 0s 4ms/step - loss: 2825.5269

Loss =  2825.52685546875

R2 =  0.5733950972173691

RMSE =  53.15568529398881

[Done] exited with code=0 in 61.233 seconds

  
