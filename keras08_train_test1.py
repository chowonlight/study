
############# < 정리한 실행 부분 > #################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([10,9,8,7,6,5,4,3,2,1])

x_train = np.array([1,2,3,4,5,6,7])  
y_train = np.array([1,2,3,4,5,6,7])

x_test= np.array([8,9,10])  
y_test= np.array([8,9,10])


model = Sequential()
model.add(Dense(14, input_dim=1))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train , epochs=5000, batch_size=4)


loss= model.evaluate(x_test, y_test)
print('loss : ',loss)  

result=model.predict([11])
print('[11]의 예측값:', result)


################ < 작업 결과 > #####################


#  Epoch 5000/5000
#  2/2 [==============================] - 0s 972us/step - loss: 1.0151e-14
#  1/1 [==============================] - 0s 91ms/step - loss: 3.0316e-13
#  loss :  3.031649096259942e-13
#  1/1 [==============================] - 0s 67ms/step
#  [11]의 예측값: [[11.000001]]


################ < 수업 내용 > #####################


#  import numpy as np
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense


#1. 데이터

#  x=np.array([1,2,3,4,5,6,7,8,9,10])
#  y=np.array([10,9,8,7,6,5,4,3,2,1])

#  x_train = np.array([1,2,3,4,5,6,7])   # 훈련 데이터 7개
#  y_train = np.array([1,2,3,4,5,6,7])

#  x_test= np.array([8,9,10])   # 테스트 데이터 3개
#  y_test= np.array([8,9,10])


#2. 모델구성

#  model = Sequential()
#  model.add(Dense(14, input_dim=1))
#  model.add(Dense(1))


#3. 컴파일,훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(x_train, y_train , epochs=5000, batch_size=4)


#4. 평가, 예측

#  loss= model.evaluate(x_test, y_test)
#  print('loss : ',loss)    # 평가까지만 한번 실행하고 로스값 확인해 보기

#  result=model.predict([11])
#  print('[11]의 예측값:', result)

# 차이점 : train과 test 데이터를 분리해서 사용

# train은 fit훈련에, 그리고 test는 평가에 사용
#
#