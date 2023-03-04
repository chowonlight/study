
############# < 정리한 실행 부분 > #################


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([1,2,3])
y = np.array([1,2,3])

model = Sequential()
model.add(Dense(3, input_dim=1))  
model.add(Dense(4))
model.add(Dense(5))  
model.add(Dense(3))     
model.add(Dense(1))           

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=10000)


loss = model.evaluate(x, y)
print('loss = ', loss)

result = model.predict([4])
print("[4]의 예측값 = ", result)


################ < 수업 내용 > #####################


#1. 데이터

#  import numpy as np

#  x = np.array([1,2,3])
#  y = np.array([1,2,3])

#2. 모델구성

#  import tensorflow as tf
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense

#  model = Sequential()
#  model.add(Dense(3, input_dim=1))   #input layer   ctre + / -> #표시
#  model.add(Dense(4))
#  model.add(Dense(5))  
#  model.add(Dense(3))     
#  model.add(Dense(1))    #output layer         

#3. 컴파일, 훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(x, y, epochs=1000)

#4. 평가, 예측
#  loss = model.evaluate(x, y)
#  print('loss = ', loss)

#  result = model.predict([4])
#  print("[4]의 예측값 = ", result)


# 결과 --> 예측값 = 3.9999993
#
# 1/1 [==============================] - 0s 77ms/step
# [4]의 예측값 =  [[3.9999993]]
# 

