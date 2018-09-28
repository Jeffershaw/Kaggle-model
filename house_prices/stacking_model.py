import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
import numpy as np
import sys
from xgboost import XGBRegressor

##########################################data reading###############################################
train_data = pd.read_csv('data/train.csv')
train_y = train_data.SalePrice
train_data = train_data.drop('SalePrice',axis=1) #split the y value
test_data = pd.read_csv('data/test.csv')
sample_num = train_data.shape[0]
############################################wash data##################################################
#drop the feature which na data is over than 1/3
na_features = [ col for col in train_data.columns 
                            if sum(train_data[col].isnull())>sample_num/2]
reduced_train_data = train_data.drop(na_features,axis=1)
reduced_test_data = test_data.drop(na_features,axis=1)

#find the missing value in col
new_reduced_train_data = reduced_train_data.copy()
new_reduced_test_data = reduced_test_data.copy()
cols_with_missing = [col for col in new_reduced_train_data.columns
                                if new_reduced_train_data[col].isnull().any()]
for col in cols_with_missing:
    new_reduced_train_data[col+'_was_missing'] = new_reduced_train_data[col].isnull()
    new_reduced_test_data[col+'_was_missing'] = new_reduced_test_data[col].isnull()

#one-hot encode
one_hot_train_data = pd.get_dummies(new_reduced_train_data)
one_hot_test_data = pd.get_dummies(new_reduced_test_data)
new_reduced_train_data, new_reduced_test_data =one_hot_train_data.align(one_hot_test_data,join='left',axis=1)
#some one-hot encoding alignment would cause na value, therefore, they should be turned into 0
new_reduced_test_data = new_reduced_test_data.fillna(0)

#apply imputer
imputer = Imputer()
print(new_reduced_train_data.shape)
new_reduced_train_data = pd.DataFrame(imputer.fit_transform(new_reduced_train_data))
print(new_reduced_test_data.shape)
new_reduced_test_data = pd.DataFrame(imputer.fit_transform(new_reduced_test_data))