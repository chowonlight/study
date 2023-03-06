
############# < 정리한 실행 부분 > #################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array(
   [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.7],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18 , 19, 20])

print(x.shape) 
print(y.shape) 

x = x.T
print(x.shape) 

model= Sequential()
model.add(Dense(3, input_dim=3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000, batch_size=32)

loss=model.evaluate(x,y)
print('loss= ', loss)

result=model.predict([[10, 1.7, 0]])
print('[10, 1.7, 0]의 예측값 = ', result)


################ < 작업 결과 > #####################


#  (3, 10)
#  (10,)
#  (10, 3)
#
#  Epoch 5000/5000
#  1/1 [==============================] - 0s 969us/step - loss: 9.0949e-13
#  1/1 [==============================] - 0s 79ms/step - loss: 1.2733e-12
#  loss=  1.273292560798056e-12
#  1/1 [==============================] - 0s 75ms/step
#  [10, 1.7, 0]의 예측값 =  [[20.]]


################ < 수업 내용 > #####################

#  import numpy as np
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense


#1. 데이터

#  x = np.array(
#     [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#      [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.7],
#      [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
#  )
#  y = np.array([11, 12, 13, 14, 15, 16, 17, 18 , 19, 20])

# x --> (3, 10) 행렬을 (10, 3)행렬로 바꾼 후 print 해서 바뀐 모습을 살펴봄

# x = x.transpose()
#  x = x.T


#2. 모델구성

#  model= Sequential()
#  model.add(Dense(3, input_dim=3))
#  model.add(Dense(5))
#  model.add(Dense(4))
#  model.add(Dense(3))
#  model.add(Dense(1))


#3. 컴파일.훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(x, y, epochs=5000, batch_size=32)


#4. 평가, 예측

#  loss=model.evaluate(x, y)
#  print('loss= ', loss)

#  result=model.predict([[10, 1.7, 0]])
#  print('[10, 1.7, 0]의 예측값 = ', result)
#
#
