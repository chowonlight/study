
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
          validation_data=(x_val, y_val), verbose=1)


loss= model.evaluate(x_test, y_test)
print('loss : ', loss) 


result=model.predict([17])
print('[17]의 예측값', result)

