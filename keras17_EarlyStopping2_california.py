
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print('\n', x.shape, y.shape, '\n')

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)


model = Sequential()
model.add(Dense(32, input_dim=8))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
es = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=30, verbose=0, validation_split=0.2, callbacks=[es])


loss = model.evaluate(x_test, y_test)
print('\nLoss = ', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('\nR2 = ', r2)


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.figure(figsize=(9,6)) 

plt.plot(hist.history['loss'], marker='.', c='red', label='Loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='Val_Loss')

plt.title('california')
plt.xlabel('epochs')
plt.ylabel('loss, val_loss')

plt.grid()
plt.legend()
plt.show()

