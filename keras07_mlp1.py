
############# < 정리한 실행 부분 > #################


import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

x = np.array(
   [[1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 2],
    [6, 1.3],     
    [7, 1.4],       
    [8, 1.5],     
    [9, 1.6],     
    [10, 1.4]]
)     

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x.shape) 
print(y.shape) 
print()

model = Sequential()
model.add(Dense(3, input_dim=2))   
model.add(Dense(5))  
model.add(Dense(4))     
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=3000, batch_size=3)

loss = model.evaluate(x, y)
print('loss = ', loss)

result = model.predict([[10, 1.4]])
print("[[10, 1.4]]의 예측값 = ", result)


################ < 작업 결과 > #####################

#  (10, 2)
#  (10,)
#
#  Epoch 3000/3000
#  4/4 [==============================] - 0s 1ms/step - loss: 9.0949e-13
#  1/1 [==============================] - 0s 107ms/step - loss: 8.1855e-13
#  loss =  8.18545209911592e-13
#  1/1 [==============================] - 0s 57ms/step
#  [[10, 1.4]]의 예측값 =  [[20.]]

################ < 수업 내용 > #####################


#  import numpy as np
#  from tensorflow.keras.models import Sequential 
#  from tensorflow.keras.layers import Dense

#1. 데이터

#  x = np.array(
#     [[1, 1],
#      [2, 1],
#      [3, 1],
#      [4, 1],
#      [5, 2],
#      [6, 1.3],     
#      [7, 1.4],       
#      [8, 1.5],     
#      [9, 1.6],     
#      [10, 1.4]]
#  )     

# 행(데이터 갯수) 무시, 열(특성) 우선

#  y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

#  print(x.shape) # (10, 2)  --> 2개의 특성을 가진 10개의 데이터
#  print(y.shape) # (10,)

#  print()  --> 한 칸 띠워 출력하기 


#2. 모델 

#  model = Sequential()
#  model.add(Dense(3, input_dim=2))   
#  model.add(Dense(5))  
#  model.add(Dense(4))     
#  model.add(Dense(1))


#3. 컴파일, 훈련 

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(x, y, epochs=3000, batch_size=3)


# 평가, 예측

#  loss = model.evaluate(x, y)
#  print('loss = ', loss)

#  result = model.predict([[10, 1.4]])
#  print("[[10, 1.4]]의 예측값 = ", result)
#
#
