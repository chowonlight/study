
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


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
model.fit(x_train, y_train, epochs=200, batch_size=10, validation_split=0.2, verbose=0)


loss = model.evaluate(x_test, y_test)
print('\nLoss = ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('\nR2 = ', r2)



################  < 작업 결과 >  ##################

[Running] python -u "c:\Users\seongja\OneDrive\바탕 화면\study\keras15_validation7_diabetes.py"

 (442, 10) (442,) 

2023-03-28 10:34:15.549910: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library ...
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

1/5 [=====>........................] - ETA: 0s - loss: 2206.0732
5/5 [==============================] - 0s 4ms/step - loss: 1999.5219

Loss =  1999.5218505859375

1/5 [=====>........................] - ETA: 0s
5/5 [==============================] - 0s 4ms/step

R2 =  0.678483024493153

[Done] exited with code=0 in 36.744 seconds

   
