
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

x =np.array(range(1,17))
y =np.array(range(1,17))

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=123, shuffle=True)


model = Sequential()
model.add(Dense(32,activation='linear', input_dim=1))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))


model.compile(loss='mse',optimizer='adam')
model.fit(x_train, y_train, epochs=1000, batch_size=100,
        validation_split=0.2, verbose=0)

print('\n', x_train, x_test)
print('', y_train, y_test,'\n')

loss = model.evaluate(x_test, y_test)
print('\nLoss = ', loss)

result =model.predict([17])
print('\nPredict Value of [17] = ', result)


################  < 작업 결과 >  ##################

[Running] python -u "c:\Users\seongja\OneDrive\바탕 화면\study\keras15_validation4_split.py"
2023-03-27 20:26:00.329870: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library ...
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

 [ 3 14 15] [ 8 11  5  1  6 10  9 12  4  2  7 16 13]
 [ 3 14 15] [ 8 11  5  1  6 10  9 12  4  2  7 16 13] 


1/1 [==============================] - ETA: 0s - loss: 4.3726e-13
1/1 [==============================] - 0s 40ms/step - loss: 4.3726e-13

Loss =  4.372570556500366e-13

1/1 [==============================] - ETA: 0s
1/1 [==============================] - 0s 484ms/step

Predict Value of [17] =  [[16.999998]]

[Done] exited with code=0 in 65.933 seconds
