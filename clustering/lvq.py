import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(10)


class LVQ:
    def __init__(self, t, eta=0.1, max_iter=400):
        # t[i]表示第i个原型向量对应的类别
        self.t = t
        # p[i]表示第i个原型向量的值
        self.p = None
        # 原型向量个数
        self.q = len(t)
        # 学习率
        self.eta = eta
        self.max_iter = max_iter
        self.C = None
        self.labels_ = None

    def fit(self, X, y):
        C = {}
        for i in range(self.q):
            C[i] = []
        self.p = np.zeros((len(self.t), X.shape[1]))

        # 初始化原型向量: 从类别标记对应的X中 随机选择1个作为初始原型向量
        for i in range(self.q):
            candidate_indices = np.where(y == self.t[i])[0]
            target_indice = random.sample(list(candidate_indices), 1)
            self.p[i] = X[target_indice]

        for _ in range(self.max_iter):
            # 随机选取样本
            j = random.sample(list(range(len(y))), 1)
            # 样本与原型向量的距离
            d = np.sqrt(np.sum((X[j] - self.p) ** 2, axis=1))
            # 与样本最近的原型向量
            i_ = np.argmin(d)
            if y[j] == t[i_]:
                self.p[i_] = self.p[i_] + self.eta * (X[j] - self.p[i_])
            else:
                self.p[i_] = self.p[i_] - self.eta * (X[j] - self.p[i_])

        # 迭代完成后，划分聚类簇
        for j in range(X.shape[0]):
            d = np.sqrt(np.sum((X[j] - self.p) ** 2, axis=1))
            i_ = np.argmin(d)
            C[i_].append(j)
        self.C = C
        self.labels_ = np.zeros((X.shape[0],), dtype=np.int32)
        for i in range(self.q):
            self.labels_[C[i]] = i

    def predict(self, X):
        pred_y = []
        for j in range(X.shape[0]):
            d = np.sqrt(np.sum((X[j] - self.p) ** 2, axis=1))
            i_ = np.argmin(d)
            pred_y.append(self.t[i_])
        return np.array(pred_y)


if __name__ == "__main__":
    import pandas as pd
    import os

    # 读取西瓜数据集4.0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../data/watermelon4.csv")
    data = pd.read_csv(file_path, header=None)
    X = data.values
    y = np.zeros((X.shape[0],), dtype=np.int32)
    y[range(9, 21)] = 1
    # 各原型向量的预设类别标记
    t = np.array([0, 1, 1, 0, 0], dtype=np.int32)

    print(y)
    X_test = X
    lvq = LVQ(t)
    lvq.fit(X, y)
    print(lvq.C)
    print(lvq.labels_)
    print(lvq.predict(X))
    # 坐标轴范围
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 0.8)
    # 坐标轴刻度
    plt.xticks([i / 10 for i in range(1, 10)])
    plt.yticks([i / 10 for i in range(9)])
    # 坐标轴名称
    plt.xlabel("density")
    plt.ylabel("sugar content")
    plt.scatter(X[:, 0], X[:, 1], c=lvq.labels_)
    plt.scatter(lvq.p[:, 0], lvq.p[:, 1], c="red", marker="+")
    plt.title("lvq")
    plt.show()
