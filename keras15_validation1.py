
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

x_train = np.array(range(1, 11))
y_train = np.array(range(1, 11))

x_val = np.array([14, 15, 16])
y_val = np.array([14, 15, 16])

x_test = np.array([11, 12, 13])
y_test = np.array([11, 12, 13])


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

1/5 [=====>........................] - ETA: 0s - loss: 1.1369e-13
5/5 [==============================] - 0s 22ms/step - loss: 2.0037e-13 - val_loss: 5.7601e-12
Epoch 250/250

1/5 [=====>........................] - ETA: 0s - loss: 2.2737e-13
5/5 [==============================] - 0s 16ms/step - loss: 2.2169e-13 - val_loss: 5.7601e-12

1/1 [==============================] - ETA: 0s - loss: 6.0633e-13
1/1 [==============================] - 0s 48ms/step - loss: 6.0633e-13

Loss =  6.063298192519884e-13

Predict Value of [17] =  [[16.999998]]

[Done] exited with code=0 in 81.097 seconds

