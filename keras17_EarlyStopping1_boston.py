
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pylab as plt


datasets = load_boston()

x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)     

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=123, test_size=0.7)


model = Sequential()
model.add(Dense(10, input_dim=13, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor = 'val_loss', patience=1000, mode='min', verbose=1, restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=10000, batch_size=16, validation_split=0.2, verbose=0, callbacks=[es])

print("=========================================")
print(hist)

print("=========================================")
print(hist.history)

print("=========================================")
print(hist.history['loss'])

print("================== val_loss ====================")
print(hist.history['val_loss'])
print("================== val_loss ====================")


loss = model.evaluate(x_test, y_test)
print('\nLoss = ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('\nR2 = ', r2)


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6))

plt.plot(hist.history['loss'], marker='.', c='red', label='Loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='Val_Loss')

plt.title('Boston')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')

plt.grid()
plt.legend()
plt.show()

