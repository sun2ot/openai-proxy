import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff


class AGNES:
    def __init__(self, k, dist_type):
        self.k = k
        self.labels_ = None
        self.C = {}
        self.dist_type = dist_type
        self.dist_func = None

        if dist_type == "MIN":
            self.dist_func = self.mindist
        elif dist_type == "MAX":
            self.dist_func = self.maxdist
        elif dist_type == "AVG":
            self.dist_func = self.avgdist
        elif dist_type == "hausdorff":
            self.dist_func = directed_hausdorff

    def fit(self, X):
        # 初始化单样本聚类簇
        for j in range(X.shape[0]):
            self.C[j] = set()
            self.C[j].add(j)
        # 初始化聚类簇距离矩阵
        # 对角线元素置为极大值(Ci-Ci的距离是0，会影响最近聚类簇的判断)
        M = 1e10 * np.ones((X.shape[0], X.shape[0]), dtype=np.float32)
        if self.dist_type == "hausdorff":
            for i in range(X.shape[0]):
                for j in range(i + 1, X.shape[0]):
                    # 豪斯多夫距离：max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
                    M[i, j] = max(
                        self.dist_func(X[list(self.C[i])], X[list(self.C[j])])[0],
                        self.dist_func(X[list(self.C[j])], X[list(self.C[i])])[0],
                    )
                    M[j, i] = M[i, j]
        else:
            print("excute else")
            for i in range(X.shape[0]):
                for j in range(i + 1, X.shape[0]):
                    M[i, j] = self.dist_func(X, self.C[i], self.C[j])
                    M[j, i] = M[i, j]
        # 初始化当前聚类簇个数
        q = X.shape[0]
        while q > self.k:
            # 多维数组中，argmin会展平再返回最小值索引
            index = np.argmin(M)
            # f1. 计算得出行列索引
            # i_ = index // M.shape[1]
            # j_ = index % M.shape[1]
            # f2. 直接返回行列索引
            index = np.unravel_index(index, M.shape)
            i_ = index[0]
            j_ = index[1]
            self.C[i_] = set(self.C[i_].union(self.C[j_]))
            # 删除C[j_]
            for j in range(j_ + 1, q):
                self.C[j - 1] = set(self.C[j])
            # 释放末位聚类簇
            del self.C[q - 1]
            # 删除距离矩阵中的j_行/列
            M = np.delete(M, j_, axis=0)
            M = np.delete(M, j_, axis=1)
            if self.dist_type == "hausdorff":
                for j in range(q - 1):
                    if i_ != j:
                        M[i_, j] = max(
                            self.dist_func(X[list(self.C[i_])], X[list(self.C[j])])[0],
                            self.dist_func(X[list(self.C[j])], X[list(self.C[i_])])[0],
                        )
                        M[j, i_] = M[i_, j]
            else:
                for j in range(q - 1):
                    if i_ != j:
                        M[i_, j] = self.dist_func(X, self.C[i_], self.C[j])
                        M[j, i_] = M[i_, j]
            q -= 1
        # 初始化标签
        self.labels_ = np.zeros((X.shape[0],), dtype=np.int32)
        for i in range(self.k):
            self.labels_[list(self.C[i])] = i

    @classmethod
    def mindist(cls, X, Ci, Cj):
        Xi = X[list(Ci)]
        Xj = X[list(Cj)]
        min = 1e10
        for i in range(len(Xi)):
            d = np.sqrt(np.sum((Xi[i] - Xj) ** 2, axis=1))
            dmin = np.min(d)
            if dmin < min:
                min = dmin
        return min

    @classmethod
    def maxdist(cls, X, Ci, Cj):
        Xi = X[list(Ci)]
        Xj = X[list(Cj)]
        max = 0
        for i in range(len(Xi)):
            d = np.sqrt(np.sum((Xi[i] - Xj) ** 2, axis=1))
            dmax = np.max(d)
            if dmax > max:
                max = dmax
        return max

    @classmethod
    def avgdist(cls, X, Ci, Cj):
        Xi = X[list(Ci)]
        Xj = X[list(Cj)]
        sum = 0.0
        for i in range(len(Xi)):
            d = np.sqrt(np.sum((Xi[i] - Xj) ** 2, axis=1))
            sum += np.sum(d)
        dist = sum / (len(Ci) * len(Cj))
        return dist


if __name__ == "__main__":
    import pandas as pd
    import os

    # 读取西瓜数据集4.0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../data/watermelon4.csv")
    data = pd.read_csv(file_path, header=None)
    X = data.values
    X_test = X

    agnes = AGNES(4, dist_type="hausdorff")
    agnes.fit(X)
    print("C:", agnes.C)
    print(agnes.labels_)
    # 坐标轴范围
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 0.8)
    # 坐标轴刻度
    plt.xticks([i / 10 for i in range(1, 10)])
    plt.yticks([i / 10 for i in range(9)])
    # 坐标轴名称
    plt.xlabel("density")
    plt.ylabel("sugar content")
    plt.scatter(X[:, 0], X[:, 1], c=agnes.labels_)
    plt.title("agens")

    plt.show()
