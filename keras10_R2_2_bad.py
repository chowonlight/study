
############# < 정리한 실행 부분 > #################


#          < 실행 및 output 조건 >

#1. R2를 음수가 아닌 0.5 이하로 만들 것

#2. 데이터는 건들지 말것

#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size = 1
#5. 히든레이어의 노드는 10개 이상 100개 이하 
#6. train 사이즈 75%
#7. epoch 100번 이상
#8. loss 지표는 mse, mae


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score


x=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y=np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])

x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.75,  
    shuffle=True,
    random_state=100)


model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(25))
model.add(Dense(15))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(25))
model.add(Dense(12))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=450, batch_size=1)


loss= model.evaluate(x_test, y_test)
print('loss : ', loss) 

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 =', r2)


################ < 작업 결과 > #####################


#  Epoch 450/450
#  15/15 [==============================] - 0s 997us/step - loss: 9.2113
#  1/1 [==============================] - 0s 117ms/step - loss: 28.5040
#  loss :  28.50398826599121
#  1/1 [==============================] - 0s 80ms/step
#  r2 = 0.018457596350956118


################ < 수업 내용 > #####################


#1. R2를 음수가 아닌 0.5 이하로 만들 것
#2. 데이터는 건들지 말것
#3. 레이어는 인풋 아웃풋 포함 7개 이상
#4. batch_size = 1
#5. 히든레이어의 노드는 10개 이상 100개 이하 
#6. train 사이즈 75%
#7. epoch 100번 이상
#8. loss 지표는 mse, mae
# [실습시작]


#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense
#  import numpy as np
#  from sklearn.model_selection import train_test_split 



#1. 데이터

#  x=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#  y=np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])

#  x_train, x_test, y_train, y_test = train_test_split(x, y, 
#      train_size=0.75,  
#      shuffle=True,
#      random_state=100)


#2. 모델 구성

#  model = Sequential()
#  model.add(Dense(10, input_dim=1))
#  model.add(Dense(25))
#  model.add(Dense(15))
#  model.add(Dense(30))
#  model.add(Dense(20))
#  model.add(Dense(30))
#  model.add(Dense(25))
#  model.add(Dense(12))
#  model.add(Dense(1))


#3. 컴파일 훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(x_train, y_train, epochs=450, batch_size=1)


#4. 평가, 예측

#  loss= model.evaluate(x_test, y_test)
#  print('loss : ', loss) 

#  y_predict = model.predict(x_test)

#  from sklearn.metrics import r2_score
#  r2 = r2_score(y_test, y_predict)
#  print('r2 =', r2)
#
#