

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split 

x_train = np.array(range(1, 17))
y_train = np.array(range(1, 17))

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=13/16, random_state=1234, shuffle=False)     
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=10/13, random_state=1234, shuffle=False)     

print('\n', x_train, x_test, x_val)
print('', y_train, y_test, y_val,'\n')

model = Sequential()
model.add(Dense(5, activation='linear', input_dim=1))
model.add(Dense(12))
model.add(Dense(24))
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=250, batch_size=2, 
            validation_data=(x_val, y_val), verbose=0)


loss= model.evaluate(x_test, y_test)
print('\nLoss = ', loss) 


result=model.predict([17])
print('\nPredict Value of [17] = ', result)



################  < 작업 결과 >  ##################

[Running] python -u "c:\Users\sengja\OneDrive\바탕 화면\study\keras15_validation3_train_test.py"

 [ 1  2  3  4  5  6  7  8  9 10] [11 12 13] [14 15 16]
 [ 1  2  3  4  5  6  7  8  9 10] [11 12 13] [14 15 16] 

2023-03-27 19:54:35.594198: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library ...
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

1/1 [==============================] - ETA: 0s - loss: 1.8190e-12
1/1 [==============================] - 0s 31ms/step - loss: 1.8190e-12

Loss =  1.8189894035458565e-12

Predict Value of [17] =  [[17.]]

[Done] exited with code=0 in 32.272 seconds

##################################################
