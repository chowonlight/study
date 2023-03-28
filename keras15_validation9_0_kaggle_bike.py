
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


path = './_data/kaggle_bike/'
path_save = './_save/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

print('\n', train_csv.shape)    
print('\n', test_csv.shape)    

print('\n', train_csv.info())
print('\n', train_csv.describe())

print('\n', train_csv.columns)
print('\n', test_csv.columns)
print('\n', type(train_csv))    



################  < 작업 결과 >  ##################

[Running] python -u "c:\Users\seongja\OneDrive\바탕 화면\study\keras15_validation9_0_kaggle_bike.py"

 (10886, 11)

 (6493, 8)
<class 'pandas.core.frame.DataFrame'>
Index: 10886 entries, 2011/01/01 0:00 to 2012/12/19 23:00
Data columns (total 11 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   season      10886 non-null  int64  
 1   holiday     10886 non-null  int64  
 2   workingday  10886 non-null  int64  
 3   weather     10886 non-null  int64  
 4   temp        10886 non-null  float64
 5   atemp       10886 non-null  float64
 6   humidity    10886 non-null  int64  
 7   windspeed   10886 non-null  float64
 8   casual      10886 non-null  int64  
 9   registered  10886 non-null  int64  
 10  count       10886 non-null  int64  
dtypes: float64(3), int64(8)
memory usage: 1020.6+ KB

 None

              season       holiday  ...    registered         count
count  10886.000000  10886.000000  ...  10886.000000  10886.000000
mean       2.506614      0.028569  ...    155.552177    191.574132
std        1.116174      0.166599  ...    151.039033    181.144454
min        1.000000      0.000000  ...      0.000000      1.000000
25%        2.000000      0.000000  ...     36.000000     42.000000
50%        3.000000      0.000000  ...    118.000000    145.000000
75%        4.000000      0.000000  ...    222.000000    284.000000
max        4.000000      1.000000  ...    886.000000    977.000000

[8 rows x 11 columns]

 Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed', 'casual', 'registered', 'count'],
      dtype='object')

 Index(['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp',
       'humidity', 'windspeed'],
      dtype='object')

 <class 'pandas.core.frame.DataFrame'>

[Done] exited with code=0 in 120.692 seconds

   
