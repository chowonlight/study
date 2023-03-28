
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(x.shape, y.shape)
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=123, shuffle=True)

model = Sequential()
model.add(Dense(32, input_dim=8))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=20, batch_size=20, validation_split=0.2, verbose=0)

print(hist.history)

plt.figure(figsize=(9,6))

plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')

plt.title('')
plt.show()



################  < 작업 결과 >  ##################


[Running] python -u "c:\Users\seongja\OneDrive\바탕 화면\study\keras16_overfit2_california.py"
(20640, 8) (20640,)
2023-03-28 13:50:29.230790: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library ...
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
{'loss': [1336.884521484375, 4.20018196105957, 6.048137664794922, 23.338422775268555, 24.491682052612305, 30.816701889038086, 18.33277130126953, 1.4706666469573975, 
1.402535080909729, 1.1710008382797241, 1.0964624881744385, 2.0562734603881836, 1.042542815208435, 1.5941603183746338, 0.8720977902412415, 0.7903141379356384, 
0.8314303755760193, 0.8851296305656433, 1.7537941932678223, 0.8081075549125671], 'val_loss': [8.316774368286133, 2.861016273498535, 1.3814976215362549, 
1.7384998798370361, 1.0596097707748413, 1.5673130750656128, 18.94059944152832, 1.0458226203918457, 1.2934589385986328, 0.8728801012039185, 1.192156434059143, 
1.0796663761138916, 1.4497973918914795, 0.9184064269065857, 0.7143864035606384, 1.3177248239517212, 1.7077183723449707, 0.9582122564315796, 1.0184746980667114, 
0.9438100457191467]}

[Done] exited with code=0 in 255.327 seconds
