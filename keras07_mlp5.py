
############# < 정리한 실행 부분 > #################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=np.array([range(10), range(21, 31), range(201, 211)]) 
w = x.T 

y=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]]) 
z = y.T 

print(x.shape)  
print(w.shape)  
print(y.shape)  
print(z.shape)  
print()   

model= Sequential()
model.add(Dense(3, input_dim=3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))

model.compile(loss='mse', optimizer='adam')
model.fit(w, z, epochs=10000, batch_size=3)

loss=model.evaluate(w, z)
print('loss= ', loss)

result=model.predict([[9, 30, 210]])
print('[9, 30, 210]의 예측값', result)

  
################ < 작업 결과 > #####################


#  (3, 10)
#  (10, 3)
#  (2, 10)
#  (10, 2)
#
#  Epoch 10000/10000
#  4/4 [==============================] - 0s 34us/step - loss: 3.8145e-09
#  1/1 [==============================] - 0s 96ms/step - loss: 4.3074e-09
#  loss=  4.307355361277132e-09
#  1/1 [==============================] - 0s 82ms/step
#  [9, 30, 210]의 예측값 [[9.999905  1.9000161]] 


################ < 수업 내용 > #####################


# x는 3개
# y는 2개

#  import numpy as np
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense


#1. 데이터

#  x=np.array([range(10), range(21, 31), range(201, 211)])    #(3, 10)

#  w = x.T   #(10, 3)


#  y=np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]])   #(2, 10)

#  z = y.T  #(10, 2)

#  print(x.shape)  
#  print(w.shape)  
#  print(y.shape)  
#  print(z.shape)  
#  print()   


# 예측 : [[9, 30, 210]] ---> 예상 : y값 10, 1.9

#2. 모델구성

#  model= Sequential()
#  model.add(Dense(3, input_dim=3))
#  model.add(Dense(5))
#  model.add(Dense(4))
#  model.add(Dense(3))
#  model.add(Dense(2))


#3. 컴파일, 훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(w, z, epochs=10000, batch_size=3)


#4. 평가, 예측

#  loss=model.evaluate(w, z)
#  print('loss= ', loss)

#  result=model.predict([[9, 30, 210]])
#  print('[9, 30, 210]의 예측값', result)


#  ---> 가시적인 확인을 위해서 다시 한 번 더 출력
#  print()    
#  print(x.shape)   #(3, 10)   
#  print(w.shape)   #(10, 3)
#  print(y.shape)   #(2, 10)
#  print(z.shape)   #(10, 2)
#  
#  