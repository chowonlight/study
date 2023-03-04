
############# < 정리한 실행 부분 > #################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array(
   [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.7]]
)
y = np.array([11, 12, 13, 14, 15, 16, 17, 18 , 19, 20])

w = x.T

print(w.shape) 
print(y.shape) 
print()

model = Sequential()
model.add(Dense(3, input_dim=2))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(w,y, epochs=7000, batch_size=3)

loss = model.evaluate(w,y)
print('loss : ', loss)

result = model.predict([[10, 1.7]])
print('[10, 1.7]의 예측값', result)


################ < 작업 결과 > #####################


#  (2, 10)
#  (10,)
#  
#  Epoch 7000/7000
#  4/4 [==============================] - 0s 1ms/step - loss: 1.2267e-08
#  1/1 [==============================] - 0s 90ms/step - loss: 9.6901e-09
#  loss :  9.690120705840854e-09
#  1/1 [==============================] - 0s 71ms/step
#  [10, 1.7]의 예측값 [[20.000025]]


################ < 수업 내용 > #####################

#  import numpy as np
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense


#1. 데이터

#  x = np.array(
#     [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#      [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.7]]
#  )
#  y = np.array([11, 12, 13, 14, 15, 16, 17, 18 , 19, 20])

# w = x.transpose()  ---> 전치 행렬 만들기
#  w = x.T

#  print(w.shape) #(2, 10) ---> 10개의 특성을 가진 2개의 데이터
#  print(y.shape) #(10,)

#  print()  ---> 한 칸 띄워 출력하기


#2. 모델구성

#  model = Sequential()
#  model.add(Dense(3, input_dim=2))
#  model.add(Dense(5))
#  model.add(Dense(4))
#  model.add(Dense(1))


#3. 컴파일,훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(w,y, epochs=7000, batch_size=3)


#4. 평가, 예측

#  loss = model.evaluate(w,y)
#  print('loss : ', loss)

#  result = model.predict([[10, 1.7]])
#  print('[10, 1.7]의 예측값', result)
#
#