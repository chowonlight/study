
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

