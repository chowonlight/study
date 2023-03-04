
############# < 정리한 실행 부분 > #################

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([1,2,3])
y = np.array([1,2,3])

model = Sequential()
model.add(Dense(1, input_dim=1))
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=4000)


################ < 작업 결과 > #####################

# 결과 --> 4,000 번째 최종 loss

# Epoch 4000/4000
# 1/1 [==============================] - 0s 2ms/step - loss: 0.0016

################ < 수업 내용 > #####################

#  import numpy as np
#  x = np.array([1,2,3])
#  y = np.array([1,2,3])

#2. 모델구성
#  import tensorflow as tf
#  from tensorflow.keras.models import Sequential
#  from tensorflow.keras.layers import Dense

#  model = Sequential()
#  model.add(Dense(1, input_dim=1))

#3. 컴파일 훈련
#  model.compile(loss='mse', optimizer='adam')
#  model.fit(x, y, epochs=4000)
# 