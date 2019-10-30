'''=================================================
@Project -> File   ：titanic -> data_processing
@Author ：Zhuang Yi
@Date   ：2019/10/24 16:09
=================================================='''

import datetime

import pandas as pd
from util import TRAIN_SET_PATH
from datetime import timedelta
from sklearn import preprocessing
import numpy as np

pd.set_option('display.max_columns', 500)


class Clean_Data(object):
    def __init__(self):
        self.train_set = pd.read_csv(TRAIN_SET_PATH)

    def show_raw_data(self):
        print(self.train_set.head(10))
        print(len(self.train_set))
        print(self.train_set.dtypes)

    def data_process(self):
        ds = self.train_set.copy()
        ds = ds.drop_duplicates()  # drop  duplicates
        average_age = 29.88
        mean_fare = 33.3
        ds.Age.fillna(value=average_age, inplace=True)
        ds.Fare.fillna(value=mean_fare, inplace=True)
        lenc = preprocessing.LabelEncoder()
        menc = preprocessing.LabelEncoder()
        lenc.fit(ds.Sex.values.tolist())
        menc.fit(ds.Embarked.values.tolist())
        ds.Sex = lenc.transform(ds.Sex.values.tolist())
        ds.Embarked = menc.transform(ds.Embarked.values.tolist())

        return ds.values

    def split_train_test(self):
        n_values = self.data_process()
        min_max_scaler = preprocessing.MinMaxScaler()
        n_values = min_max_scaler.fit_transform(n_values)
        train = n_values[:891, :]
        np.random.shuffle(train)
        test = n_values[891:, :]
        train_x, train_l = train[:, 1:-2], train[:, -1]
        test_x = test[:, 1:-2]
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
        return train_x, train_l, test_x

    def reverse_train_test(self):
        train_x, train_l, test_x = self.split_train_test()
        train_x = np.reshape(train_x, (-1, 6))
        test_x = np.reshape(test_x, (-1, 6))
        for i in range(6):
            temp = train_x[:, i]
            train_x[:, i] = train_x[:, 5-i]
            train_x[:, 5-i] = temp
            temo = test_x[:, i]
            test_x[:, i] = test_x[:, 5 - i]
            test_x[:, 5 - i] = temo
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
        return train_x, test_x

    def split_train_test_mlp(self):
        n_values = self.data_process()
        min_max_scaler = preprocessing.MinMaxScaler()
        n_values = min_max_scaler.fit_transform(n_values)
        train = n_values[:891, :]
        np.random.shuffle(train)
        test = n_values[891:, :]
        train_x, train_l = train[:, 1:-1], train[:, -1]
        test_x = test[:, 1:-1]
        print(train_x)
        return train_x, train_l, test_x


if __name__ == '__main__':
    cd = Clean_Data()
    cd.split_train_test_mlp()
