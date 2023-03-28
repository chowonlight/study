
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

