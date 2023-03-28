
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print('\n', x.shape, y.shape, '\n')   

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123)

model = Sequential()
model.add(Dense(32,input_dim=8))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=30, validation_split=0.2, verbose=0)

loss = model.evaluate(x_test, y_test)
print('\nLoss = ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('\nR2 = ', r2)



################  < 작업 결과 >  ##################

