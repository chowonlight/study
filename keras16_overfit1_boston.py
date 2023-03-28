
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pylab as plt


datasets = load_boston()

x = datasets['data']
y = datasets['target']

print('\n', x.shape, y.shape, '\n')   

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, test_size=0.7)


model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))


model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=10, batch_size=8, validation_split=0.2, verbose=0)

print('\n', hist, '\n') 
print()
print(hist.history) 
print()
print('\n', hist.history['loss']) 
print()
print(hist.history['val_loss']) 


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')  

plt.title('boston')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')

plt.grid()
plt.legend()
plt.show()

