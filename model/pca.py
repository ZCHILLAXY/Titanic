'''=================================================
@Project -> File   ：titanic -> pca
@Author ：Zhuang Yi
@Date   ：2019/10/25 15:39
=================================================='''

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from data_processing.data_processing import Clean_Data


class my_PCA(object):
    def __init__(self):
        self.sd = Clean_Data()
        self.training_x, self.training_y, self.testing_x = self.sd.split_train_test()

    def calculate_pca(self, sample):
        y = np.reshape(self.training_y, (-1, 1))
        x = np.reshape(self.training_x, (-1, 7))
        pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
        reduced_x = pca.fit_transform(x)  # 对样本进行降维
        print(reduced_x.shape)
        reduced_sample = pca.fit_transform(sample)
        return reduced_sample
        # print(reduced_x)
        #
        # red_x, red_y = [], []
        # blue_x, blue_y = [], []
        # green_x, green_y = [], []
        #
        # for i in range(len(reduced_x)):
        #     if y[i] == 0:
        #         red_x.append(reduced_x[i][0])
        #         red_y.append(reduced_x[i][1])
        #
        #     elif y[i] == 1:
        #         blue_x.append(reduced_x[i][0])
        #         blue_y.append(reduced_x[i][1])
        #
        #     else:
        #         green_x.append(reduced_x[i][0])
        #         green_y.append(reduced_x[i][1])
        #
        # plt.scatter(red_x, red_y, c='r', marker='x')
        # plt.scatter(blue_x, blue_y, c='b', marker='.')
        # plt.scatter(green_x, green_y, c='g', marker='.')
        # plt.show()


if __name__ == '__main__':
    pca = my_PCA()
    pca.calculate_pca()
