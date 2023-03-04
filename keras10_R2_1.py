
############# < 정리된 실행 부분 > #################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score


x=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
y=np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])

x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.8,  
    shuffle=True,
    random_state=150)

model = Sequential()
model.add(Dense(3, input_dim=1))
model.add(Dense(7))
model.add(Dense(15))
model.add(Dense(7))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=6000, batch_size=3)


loss= model.evaluate(x_test, y_test)
print('loss : ', loss) 

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 =', r2)


################ < 작업 결과 > #####################


#  Epoch 6000/6000
#  6/6 [==============================] - 0s 1ms/step - loss: 12.3816
#  1/1 [==============================] - 0s 113ms/step - loss: 5.8094
#  loss :  5.80940055847168
#  1/1 [==============================] - 0s 70ms/step
#  r2 = 0.85721904888589


################ < 수업 내용 > #####################

#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense
#  import numpy as np
#  from sklearn.model_selection import train_test_split 


#1. 데이터

#  x=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
#  y=np.array([1, 2, 4, 3, 5, 7, 9, 3, 8, 12, 13, 8, 14, 15, 9, 6, 17, 23, 21, 20])

#  x_train, x_test, y_train, y_test = train_test_split(x, y, 
#      train_size=0.8,  
#      shuffle=True,
#      random_state=150)


#2. 모델 구성

#  model = Sequential()
#  model.add(Dense(3, input_dim=1))
#  model.add(Dense(7))
#  model.add(Dense(15))
#  model.add(Dense(7))
#  model.add(Dense(1))


#3. 컴파일 훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(x_train, y_train, epochs=6000, batch_size=3)


#4. 평가, 예측

#  loss= model.evaluate(x_test, y_test)
#  print('loss : ', loss) 

#  y_predict = model.predict(x_test)

#  from sklearn.metrics import r2_score

#  r2 = r2_score(y_test, y_predict)
#  print('r2 =', r2)
#
#