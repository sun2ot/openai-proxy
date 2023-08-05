from typing import Set, Any

import numpy as np
import matplotlib.pyplot as plt
import random
from queue import Queue

# 固定一个随机序列
random.seed(1)


class DBSCAN:
    def __init__(self, epsilon=0.11, min_pts=5):
        """
        :param epsilon: 距离参数
        :param min_pts: 构成核心对象的最小样本数
        """
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.labels = None  # 簇标记
        self.C = None  # 簇
        self.Omega = set()  # 核心对象集合
        self.N_epsilon = {}  # epsilon 邻域集合

    # p213 图9.9 DBSCAN算法
    def fit(self, X):
        """
        :param X: 样本集
        """

        # 初始化簇划分为空集
        self.C = {}
        for j in range(X.shape[0]):
            # 计算 epsilon 距离 (此处为L2距离)
            dist = np.sqrt(np.sum((X - X[j]) ** 2, axis=1))
            # 确定样本 Xj 的 epsilon 邻域集
            self.N_epsilon[j] = np.where(dist <= self.epsilon)[0]
            if len(self.N_epsilon[j]) >= self.min_pts:
                # 样本加入核心对象集合
                self.Omega.add(j)

        # 初始化聚类簇数
        k = 0
        # 初始化未访问样本集合
        _Gamma = set(range(X.shape[0]))

        while len(self.Omega) > 0:
            # 记录当前未访问样本集合
            old_Gamma = set(_Gamma)
            # 随机选取一个核心对象
            # todo: omega转为list是否必须？
            o = random.sample(list(self.Omega), 1)[0]
            # 初始化队列 Q
            Q = Queue()
            Q.put(o)
            # 减去已遍历样本
            _Gamma.remove(o)
            while not Q.empty():
                q = Q.get()
                if len(self.N_epsilon[q]) >= self.min_pts:
                    # 如果为核心对象，则找出其直接密度可达的样本
                    _Delta = set(self.N_epsilon[q]).intersection(set(_Gamma))
                    for delta in _Delta:
                        # 将 _Delta 中的样本加入队列 Q
                        Q.put(delta)
                        # 减去已遍历样本
                        _Gamma.remove(delta)
            # 生成聚类簇
            self.C[k] = old_Gamma.difference(_Gamma)
            # 更新未访问样本集合
            self.Omega = self.Omega.difference(self.C[k])
            k += 1

        # 初始化簇标记
        self.labels = np.zeros((X.shape[0],), dtype=np.int32)
        for j in range(k):
            temp = list(self.C[j])
            self.labels[list(self.C[j])] = j


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("wl4.csv", header=None)
    Y = data.values
    X = np.array(
        [
            [0.697, 0.460],
            [0.774, 0.376],
            [0.634, 0.264],
            [0.608, 0.318],
            [0.556, 0.215],
            [0.403, 0.237],
            [0.481, 0.149],
            [0.437, 0.211],
            [0.666, 0.091],
            [0.243, 0.267],
            [0.245, 0.057],
            [0.343, 0.099],
            [0.639, 0.161],
            [0.657, 0.198],
            [0.360, 0.370],
            [0.593, 0.042],
            [0.719, 0.103],
            [0.359, 0.188],
            [0.339, 0.241],
            [0.282, 0.257],
            [0.748, 0.232],
            [0.714, 0.346],
            [0.483, 0.312],
            [0.478, 0.437],
            [0.525, 0.369],
            [0.751, 0.489],
            [0.532, 0.472],
            [0.473, 0.376],
            [0.725, 0.445],
            [0.446, 0.459],
        ]
    )

    dbscan = DBSCAN()
    dbscan.fit(X)
    print("C:\n", dbscan.C)
    print("labels:\n", dbscan.labels)
    plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels)
    plt.xlim(0.1, 0.9)  # 设置x轴范围为0到0.9
    plt.ylim(0, 0.8)  # 设置y轴范围为0到0.9
    plt.xticks([i / 10 for i in range(1, 10)])  # 设置x轴刻度分割点为0.0, 0.1, 0.2, ..., 0.9
    plt.yticks([i / 10 for i in range(9)])  # 设置y轴刻度分割点为0.0, 0.1, 0.2, ..., 0.9
    # plt.figure(12)
    # plt.subplot(121)

    plt.title("tinyml")

    # import sklearn.cluster as cluster
    # sklearn_DBSCAN=cluster.DBSCAN(eps=0.11,min_samples=5,metric='l2')
    # sklearn_DBSCAN.fit(X)
    # print(sklearn_DBSCAN.labels_)
    # plt.subplot(122)
    # plt.scatter(X[:,0],X[:,1],c=sklearn_DBSCAN.labels_)
    # plt.title('sklearn')
    plt.show()
