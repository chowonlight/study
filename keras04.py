
############# < 정리한 실행 부분 = 수업내용 > #################

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 5, 4])

model = Sequential()
model.add(Dense(5, input_dim=1))   
model.add(Dense(8))  
model.add(Dense(10))     
model.add(Dense(8)) 
model.add(Dense(5)) 
model.add(Dense(1)) 

model.compile(loss='mae', optimizer='adam')
model.fit(x, y, epochs=7000)

loss = model.evaluate(x, y)
print('loss = ', loss)

result = model.predict([6])
print("[6]의 예측값 = ", result)


################ < 작업 결과 > #####################

#  Epoch 7000/7000
#  1/1 [==============================] - 0s 4ms/step - loss: 0.4014
#  1/1 [==============================] - 0s 100ms/step - loss: 0.4015
#  loss =  0.4015394151210785
#  1/1 [==============================] - 0s 74ms/step
#  [6]의 예측값 =  [[5.993847]]
#
