
############# < 정리한 실행 부분 > #################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x=np.array([range(10), range(21, 31), range(201, 211)]) 
print(x.shape)  

x = x.T
print(x.shape)  

y=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]) 
print(y.shape)  


y = y.T 
print(y.shape)  

model = Sequential()
model.add(Dense(3, input_dim=3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=7000, batch_size=3)

loss=model.evaluate(x, y)
print('loss= ', loss)

result=model.predict([[9, 30, 210]])
print('[9, 30, 210]의 예측값', result)


################ < 작업 결과 > #####################


#  (3, 10)
#  (10, 3)
#  (3, 10)
#  (10, 3)
#  
#  Epoch 7000/7000
#  4/4 [==============================] - 0s 2ms/step - loss: 1.1250e-09
#  1/1 [==============================] - 0s 115ms/step - loss: 4.3584e-09
#  loss=  4.358437166729345e-09
#  1/1 [==============================] - 0s 77ms/step
#  [9, 30, 210]의 예측값 [[ 9.9999704e+00  1.9000973e+00 -1.3605412e-04]]


################ < 수업 내용 > #####################


# x는 3개
# y는 3개

#  import numpy as np
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense


#1. 데이터

#  x=np.array([range(10), range(21, 31), range(201, 211)])    #(3, 10)

#  w = x.T   #(10, 3)


#  y=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
#              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])     #(3, 10)

#  z = y.T    #(10, 3)


#  print(x.shape)  
#  print(w.shape)  
#  print(y.shape)  
#  print(z.shape)  
#  print()   


#2. 모델구성

#  model = Sequential()
#  model.add(Dense(3, input_dim=3))
#  model.add(Dense(5))
#  model.add(Dense(4))
#  model.add(Dense(3))
#  model.add(Dense(3))


#3. 컴파일, 훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(w, z, epochs=7000, batch_size=3)


#4. 평가, 예측

#  loss=model.evaluate(w, z)
#  print('loss= ', loss)
#  result=model.predict([[9, 30, 210]])
#  print('[9, 30, 210]의 예측값', result)


#  ---> 가시적인 확인을 위해서 다시 한 번 더 출력
#  print()    
#  print(x.shape)   #(3, 10)   
#  print(w.shape)   #(10, 3)
#  print(y.shape)   #(3, 10)
#  print(z.shape)   #(10, 3)
#
#
