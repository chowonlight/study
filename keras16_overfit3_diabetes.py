
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

datasets = load_diabetes()

x = datasets.data
y = datasets.target

print('\n', x.shape, y.shape, '\n')    

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=20580)


model = Sequential()
model.add(Dense(64, input_dim=10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2, verbose=0)

print(hist.history)

plt.plot(hist.history['loss'])
plt.show()

