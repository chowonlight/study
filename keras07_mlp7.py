
############# < 정리된 실행 부분 > #################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=np.array([range(10)]) 
w = x.T  

y=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])  
z = y.T 
 
print(x.shape)   
print(w.shape)  
print(y.shape) 
print(z.shape)  
print()   


model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(3))

model.compile(loss='mse', optimizer='adam')
model.fit(w, z, epochs=3000, batch_size=3)


loss=model.evaluate(w, z)
print('loss= ', loss)

result=model.predict([[9]])
print('[9]의 예측값', result)


################ < 작업 결과 > #####################


#  (1, 10)
#  (10, 1)
#  (3, 10)
#  (10, 3)
#  
#  Epoch 3000/3000
#  4/4 [==============================] - 0s 998us/step - loss: 4.9257e-10
#  1/1 [==============================] - 0s 104ms/step - loss: 4.9659e-10
#  loss=  4.965874378370927e-10
#  1/1 [==============================] - 0s 74ms/step
#  [9]의 예측값 [[1.0000008e+01 1.9000118e+00 5.7846308e-05]]


################ < 수업 내용 > #####################


# x는 1개
# y는 3개

#  import numpy as np
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense


#1. 데이터

#  x=np.array([range(10)])     #(1, 10)  

#  w = x.T      #(10, 1)


#  y=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
#              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])          #(3, 10) 

#  z = y.T    #(10, 3)
 

#  print(x.shape)    #(1, 10)  ->   #[실습]  (1, 10)을 (10, 1)로 바꿔 보자
#  print(w.shape)    #(10, 1)
#  print(y.shape)    #(3, 10)
#  print(z.shape)    #(10, 3)
#  print()   


#2. 모델구성

#  model = Sequential()
#  model.add(Dense(3, input_dim=1))
#  model.add(Dense(5))
#  model.add(Dense(4))
#  model.add(Dense(3))
#  model.add(Dense(3))


#3. 컴파일, 훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(w, z, epochs=3000, batch_size=3)


#4. 평가, 예측

#  loss=model.evaluate(w, z)
#  print('loss= ', loss)

#  result=model.predict([[9]])
#  print('[9]의 예측값', result)


#  ---> 가시적인 확인을 위해서 다시 한 번 더 출력
#  print()    
#  print(x.shape)   #(1, 10)   
#  print(w.shape)   #(10, 1)
#  print(y.shape)   #(3, 10)
#  print(z.shape)   #(10, 3)
#
#