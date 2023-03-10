
############# < 정리된 실행 부분 > #################

# ( 수행 결과 )   R2 0.62 이상

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.9,  
    shuffle=True,
    random_state=30000)

model = Sequential()
model.add(Dense(12, input_dim=10))
model.add(Dense(24))
model.add(Dense(36))
model.add(Dense(64))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=180, batch_size=8)

loss= model.evaluate(x_test, y_test)
print('loss : ', loss) 

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 =', r2)


################ < 작업 결과 > #####################

#  Epoch 180/180
#  50/50 [==============================] - 0s 1ms/step - loss: 3005.3840
#  2/2 [==============================] - 0s 0s/step - loss: 2430.2087
#  loss :  2430.208740234375
#  2/2 [==============================] - 0s 0s/step
#  r2 = 0.6267363775243255

################ < 수업 내용 > #####################

#  import numpy as np

#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense
#  from sklearn.datasets import load_diabetes
#  from sklearn.model_selection import train_test_split


#1. 데이터

#  datasets = load_diabetes()
#  x = datasets.data
#  y = datasets.target


# print(x.shape, y.shape)   # (442, 10) (442,)


############## [실습] ########
# R2 0.62 이상
#############################



#  x_train, x_test, y_train, y_test = train_test_split(x, y, 
#      train_size=0.9,  
#      shuffle=True,
#      random_state=30000)


#2. 모델 구성

#  model = Sequential()
#  model.add(Dense(12, input_dim=10))
#  model.add(Dense(24))
#  model.add(Dense(36))
#  model.add(Dense(64))
#  model.add(Dense(100))
#  model.add(Dense(80))
#  model.add(Dense(40))
#  model.add(Dense(20))
#  model.add(Dense(10))
#  model.add(Dense(1))


#3. 컴파일 훈련

#  model.compile(loss='mse', optimizer='adam')
#  model.fit(x_train, y_train, epochs=180, batch_size=8)


#4. 평가, 예측

#  loss= model.evaluate(x_test, y_test)
#  print('loss : ', loss) 

#  y_predict = model.predict(x_test)

#  from sklearn.metrics import r2_score
#  r2 = r2_score(y_test, y_predict)
#  print('r2 =', r2)
#  
#  
