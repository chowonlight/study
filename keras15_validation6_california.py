
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
model.add(Dense(86))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=30, validation_split=0.2, verbose=0)

loss = model.evaluate(x_test, y_test)
print('\nLoss = ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('\nR2 = ', r2)



################  < 작업 결과 >  ##################

[Running] python -u "c:\Users\seongja\OneDrive\바탕 화면\study\keras15_validation6_california.py"

 (20640, 8) (20640,) 

2023-03-28 09:57:03.901373: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library ...
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.

  1/194 [..............................] - ETA: 6s - loss: 0.6671
  2/194 [..............................] - ETA: 32s - loss: 0.5144
 22/194 [==>...........................] - ETA: 1s - loss: 0.6556 
 52/194 [=======>......................] - ETA: 0s - loss: 0.9923
 84/194 [===========>..................] - ETA: 0s - loss: 0.8450
108/194 [===============>..............] - ETA: 0s - loss: 0.8032
127/194 [==================>...........] - ETA: 0s - loss: 0.7571
149/194 [======================>.......] - ETA: 0s - loss: 0.7314
169/194 [=========================>....] - ETA: 0s - loss: 0.7116
194/194 [==============================] - 1s 3ms/step - loss: 0.6942

Loss =  0.6942345499992371

R2 =  0.47497506027938685

[Done] exited with code=0 in 881.568 seconds

   
