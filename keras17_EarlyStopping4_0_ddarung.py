
import pandas as pd


path = './_data/ddarung/'
path_save = './_save/ddarung/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)


print(train_csv.shape, test_csv.shape)
print(train_csv.columns, test_csv.columns)
print(train_csv.info(), test_csv.info())
print(train_csv.describe(), test_csv.describe())
print(type(train_csv), type(test_csv))

print(train_csv.isnull().sum())
train_csv = train_csv.dropna()
print(train_csv.isnull().sum())



#-------------------- Result of Work ------------------


[Running] python -u "c:\Users\sea bongja\OneDrive\바탕 화면\study\keras17_EarlyStopping4_0_ddarung.py"
(1459, 10) (715, 9)
Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
      dtype='object') Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
       'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
       'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5'],
      dtype='object')
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1459 entries, 3 to 2179
Data columns (total 10 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   hour                    1459 non-null   int64  
 1   hour_bef_temperature    1457 non-null   float64
 2   hour_bef_precipitation  1457 non-null   float64
 3   hour_bef_windspeed      1450 non-null   float64
 4   hour_bef_humidity       1457 non-null   float64
 5   hour_bef_visibility     1457 non-null   float64
 6   hour_bef_ozone          1383 non-null   float64
 7   hour_bef_pm10           1369 non-null   float64
 8   hour_bef_pm2.5          1342 non-null   float64
 9   count                   1459 non-null   int64  
dtypes: float64(8), int64(2)
memory usage: 125.4 KB
<class 'pandas.core.frame.DataFrame'>
Int64Index: 715 entries, 0 to 2177
Data columns (total 9 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   hour                    715 non-null    int64  
 1   hour_bef_temperature    714 non-null    float64
 2   hour_bef_precipitation  714 non-null    float64
 3   hour_bef_windspeed      714 non-null    float64
 4   hour_bef_humidity       714 non-null    float64
 5   hour_bef_visibility     714 non-null    float64
 6   hour_bef_ozone          680 non-null    float64
 7   hour_bef_pm10           678 non-null    float64
 8   hour_bef_pm2.5          679 non-null    float64
dtypes: float64(8), int64(1)
memory usage: 55.9 KB
None None
              hour  hour_bef_temperature  ...  hour_bef_pm2.5        count
count  1459.000000           1457.000000  ...     1342.000000  1459.000000
mean     11.493489             16.717433  ...       30.327124   108.563400
std       6.922790              5.239150  ...       14.713252    82.631733
min       0.000000              3.100000  ...        8.000000     1.000000
25%       5.500000             12.800000  ...       20.000000    37.000000
50%      11.000000             16.600000  ...       26.000000    96.000000
75%      17.500000             20.100000  ...       37.000000   150.000000
max      23.000000             30.000000  ...       90.000000   431.000000

[8 rows x 10 columns]              hour  hour_bef_temperature  ...  hour_bef_pm10  hour_bef_pm2.5
count  715.000000            714.000000  ...     678.000000      679.000000
mean    11.472727             23.263305  ...      36.930678       24.939617
std      6.928427              4.039645  ...      12.641503       10.075857
min      0.000000             14.600000  ...       9.000000        7.000000
25%      5.500000             20.300000  ...      28.000000       17.000000
50%     11.000000             22.900000  ...      35.000000       24.000000
75%     17.000000             26.375000  ...      45.000000       31.000000
max     23.000000             33.800000  ...      94.000000       69.000000

[8 rows x 9 columns]
<class 'pandas.core.frame.DataFrame'> <class 'pandas.core.frame.DataFrame'>
hour                        0
hour_bef_temperature        2
hour_bef_precipitation      2
hour_bef_windspeed          9
hour_bef_humidity           2
hour_bef_visibility         2
hour_bef_ozone             76
hour_bef_pm10              90
hour_bef_pm2.5            117
count                       0
dtype: int64
hour                      0
hour_bef_temperature      0
hour_bef_precipitation    0
hour_bef_windspeed        0
hour_bef_humidity         0
hour_bef_visibility       0
hour_bef_ozone            0
hour_bef_pm10             0
hour_bef_pm2.5            0
count                     0
dtype: int64

[Done] exited with code=0 in 29.063 seconds

  