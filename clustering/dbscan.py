import numpy as np
import matplotlib.pyplot as plt
import random
from queue import Queue


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
            # 创建当前未访问样本集合副本
            old_Gamma = set(_Gamma)
            # 随机选取一个核心对象
            # todo: omega转为list是否必须？
            # re: 必须。python 3.9后已弃用从 set() 中取样的方法，必须为 sequence 类型
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
    import os

    # 读取西瓜数据集4.0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../data/watermelon4.csv")
    data = pd.read_csv(file_path, header=None)
    X = data.values

    dbscan = DBSCAN()
    dbscan.fit(X)
    print("C:\n", dbscan.C)
    print("labels:\n", dbscan.labels)

    plt.subplots(1, 2, figsize=(12, 6))

    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=dbscan.labels)
    # 坐标轴范围
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 0.8)
    # 坐标轴刻度
    plt.xticks([i / 10 for i in range(1, 10)])
    plt.yticks([i / 10 for i in range(9)])
    # 坐标轴名称
    plt.xlabel("density")
    plt.ylabel("sugar content")
    # 图片标题
    plt.title("DBSCAN")

    import sklearn.cluster as cluster

    sklearn_DBSCAN = cluster.DBSCAN(eps=0.11, min_samples=5, metric="l2")
    sklearn_DBSCAN.fit(X)
    print("sklearn-labels:\n", sklearn_DBSCAN.labels_)

    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=sklearn_DBSCAN.labels_)
    # 与第一个子图保持一致
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 0.8)
    plt.xticks([i / 10 for i in range(1, 10)])
    plt.yticks([i / 10 for i in range(9)])
    plt.xlabel("density")
    plt.ylabel("sugar content")
    plt.title("sklearn-DBSCAN")

    plt.show()
