import pandas as pd
from sklearn.preprocessing import Imputer

train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')
one_hot_train_data = pd.get_dummies(train_data)
one_hot_test_data = pd.get_dummies(test_data)
final_train, final_test =one_hot_train_data.align(one_hot_test_data,join='left',axis=1)
new_data = train_data.copy()
cols_missing=(col for col in new_data.columns
                            if new_data[col].isnull().any())
for col in cols_missing:
    new_data[col+'_was_missing']=new_data[col].isnull()
one_hot_new_data = pd.get_dummies(new_data)
my_imputer = Imputer()
new_data = my_imputer.fit_transform(one_hot_new_data)

