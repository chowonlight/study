
############# < 정리한 실행 부분 > #################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y=np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

x_train = x[:7]   
y_train = y[:7]   

x_test = x[7:]  
y_test = y[7:]    

print(x_train.shape, x_test.shape)   
print(y_train.shape, y_test.shape)   
print()
print(x_train)   
print(y_train)   
print()
print(x_test)   
print(y_test)   
print()

model= Sequential()
model.add(Dense(7, input_dim=1))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train , epochs=2000, batch_size=4)

loss= model.evaluate(x_test, y_test)
print('loss : ', loss) 

result=model.predict([11])  
print('[11]의 예측값:', result)


################ < 작업 결과 > #####################


#  (7,) (3,)
#  (7,) (3,) 
#  
#  [1 2 3 4 5 6 7]
#  [10  9  8  7  6  5  4]
#  
#  [ 8  9 10]
#  [3 2 1]
#  
#  Epoch 2000/2000
#  2/2 [==============================] - 0s 2ms/step - loss: 1.2234e-08
#  1/1 [==============================] - 0s 106ms/step - loss: 4.5840e-08
#  loss :  4.5840391038609596e-08
#  1/1 [==============================] - 0s 58ms/step
#  [11]의 예측값: [[0.00031048]]


################ < 수업 내용 > #####################


#  import numpy as np
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense

#1. 데이터
#  x=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#  y=np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

# [실습]  넘파이 리스트의 슬라이싱 7:3으로 잘라라

#  x_train = x[0:7]
#  x_train = x[:7]   ---> # 파이썬에서 숫자는 < 0부터 시작 >이라, 0을 빼고 적는게 깔끔함
#  y_train = y[:7]   

#  x_test = x[7:10] 
#  x_test = x[7:]    # [8, 9, 10]   ---> < 끝이 10이기 때문에 10을 쓰지 않으며, 빼고 적는게 깔끔함
#  y_test = y[7:]    # [3, 2, 1]

# (7,)(3,)로 프린트 되는지 확인해보고 x_train과 / x_test값도 각각 프린트하여 
# [1, 2, 3, 4, 5, 6, 7]과 / [8, 9, 10]으로 프린트 되는지 확인해보기

#  print(x_train.shape, x_test.shape)   # (7,) (3,)
#  print(y_train.shape, y_test.shape)   # (7,) (3,)
#  print()
#  print(x_train)    # [1, 2, 3, 4, 5, 6, 7]
#  print(y_train)    # [10, 9, 8, 7, 6, 5, 4]
#  print()
#  print(x_test)     # [8, 9, 10]
#  print(y_test)     # [3, 2, 1]
#  print()


#2. 모델구성

#  model= Sequential()
#  model.add(Dense(7, input_dim=1))
#  model.add(Dense(1))


#3. 컴파일, 훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(x_train, y_train , epochs=2000, batch_size=4)


#4. 평가, 예측

#  loss= model.evaluate(x_test, y_test)
#  print('loss : ', loss) 

#  result=model.predict([11])  # ----> 예측값은 0( zero )
#  print('[11]의 예측값:', result)


#  ---> 확인을 위해서 다시 프린트 함
#  print()
#  print(x_train.shape, x_test.shape)   # (7,)(3,)   
#  print(y_train.shape, y_test.shape)   # (7,)(3,)
#  print()
#  print(x_train)   
#  print(y_train)   
#  print()
#  print(x_test)   
#  print(y_test)   
#  print()

#
# 문제점 : 전체 데이터 범위 <외부의 데이터를 예측> 할 때 ---> 오차가 큼
# 전체 데이터를 섞은 후, 랜덤하게 70% 뽑는다(train) 
# 나머지 30% (test)를 잡은 후
# 범위 외부의 데이터(실제로는 남은 내부 데이터)를 예측하게 되면, 
# 오차가 많이 줄어든다
#
#