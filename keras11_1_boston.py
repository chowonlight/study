
############# < 정리된 실행 부분 > #################

#  (수행 조건)   1. train 0.7
#  (수행 과제)   2. R2 0.8 이상

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
    train_size=0.7,  
    shuffle=True,
    random_state=800)

model = Sequential()
model.add(Dense(8, input_dim=13))
model.add(Dense(24))
model.add(Dense(36))
model.add(Dense(48))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=850, batch_size=10)

loss= model.evaluate(x_test, y_test)
print('loss : ', loss) 

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 =', r2)


################ < 작업 결과 > #####################


#  Epoch 850/850
#  36/36 [==============================] - 0s 1ms/step - loss: 3.2927
#  5/5 [==============================] - 0s 4ms/step - loss: 3.3067
#  loss :  3.306687116622925
#  5/5 [==============================] - 0s 0s/step
#  r2 = 0.743070950538335


################ < 수업 내용 > #####################

#  import numpy as np

#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense
#  from sklearn.datasets import load_boston
#  from sklearn.model_selection import train_test_split
#  from sklearn.metrics import r2_score


#1. 데이터

#  datasets = load_boston()
#  x = datasets.data
#  y = datasets.target

# print(x)
# print(y)

# print(datasets)

# print(datasets.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT']

# print(datasets.DESCR)

# print(x.shape, y.shape)   # (506, 13) (506,)


############## [실습] ########
#1. train 0.7
#2. R2 0.8 이상
#############################


#  x_train, x_test, y_train, y_test = train_test_split(x, y, 
#      train_size=0.7,  
#      shuffle=True,
#      random_state=800)


#2. 모델 구성

#  model = Sequential()
#  model.add(Dense(8, input_dim=13))
#  model.add(Dense(24))
#  model.add(Dense(36))
#  model.add(Dense(48))
#  model.add(Dense(64))
#  model.add(Dense(32))
#  model.add(Dense(16))
#  model.add(Dense(8))
#  model.add(Dense(1))


#3. 컴파일 훈련

#  model.compile(loss='mae', optimizer='adam')
#  model.fit(x_train, y_train, epochs=850, batch_size=10)


#4. 평가, 예측

#  loss= model.evaluate(x_test, y_test)
#  print('loss : ', loss) 

#  y_predict = model.predict(x_test)

#  r2 = r2_score(y_test, y_predict)
#  print('r2 =', r2)
#
#
