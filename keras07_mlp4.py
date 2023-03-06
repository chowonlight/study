
############# < 정리한 실행 부분 > #################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=np.array([range(10), range(21, 31), range(201, 211)])
print(x.shape)
                                           
x = x.T  
print(x.shape)
                                           
y=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])  
print(y.shape)

y = y.T  
print(y.shape)  

model= Sequential()
model.add(Dense(3, input_dim=3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=5000, batch_size=3)

loss=model.evaluate(x, y)
print('loss= ', loss)

result=model.predict([[9, 30, 210]])
print('[9, 30, 210]의 예측값 = ', result)
   

################ < 작업 결과 > #####################


#  (3, 10)
#  (10, 3)
#  (1, 10)
#  (10, 1)
#  
#  Epoch 5000/5000
#  4/4 [==============================] - 0s 998us/step - loss: 1.2170e-09
#  1/1 [==============================] - 0s 117ms/step - loss: 3.4015e-11
#  loss=  3.40151032340863e-11
#  1/1 [==============================] - 0s 79ms/step
#  [9, 30, 210]의 예측값 =  [[9.999995]]


################ < 수업 내용 > #####################

#  
# x는 3개
# y는 1개
#  
#  import numpy as np
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense


#1. 데이터

#  x=np.array([range(10), range(21, 31), range(201, 211)])  
#  print(x)
#  print(x.shape)   #(3, 10)

#  x = x.T   
#  print(x.shape) #(10, 3)

#  y=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])   
#  print(y.shape)   #(1, 10)

#  y = y.T   
#  print(y.shape)   #(10, 1)


#2. 모델구성

#  model= Sequential()
#  model.add(Dense(3, input_dim=3))
#  model.add(Dense(5))
#  model.add(Dense(4))
#  model.add(Dense(3))
#  model.add(Dense(1))


#3. 컴파일, 훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(x, y, epochs=5000, batch_size=3)


#4. 평가, 예측

#  loss=model.evaluate(x, y)
#  print('loss= ', loss)

#  result=model.predict([[9, 30, 210]])
#  print('[9, 30, 210]의 예측값 = ', result)
#   
#
