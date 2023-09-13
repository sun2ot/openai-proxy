import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(1)


class KMeans:
    def __init__(self, k=3):
        self.labels_ = None
        self.mu = None
        self.k = k

    def init(self, X, method="kmeans++", random_state=False):
        if method == "kmeans++":
            if random_state is False:
                np.random.seed(0)
            # 随机选择一个样本点作为第一个初始化的聚类中心
            mus = [X[np.random.randint(0, len(X))]]
            while len(mus) < self.k:
                Dxs = []
                array_mus = np.array(mus)
                for x in X:
                    Dx = np.min(np.sqrt(np.sum((x - array_mus) ** 2, axis=1)))
                    Dxs.append(Dx)
                Dxs = np.array(Dxs)
                # 距离最大的点，被选为聚类中心
                index = np.argmax(Dxs)
                mus.append(X[index])
            self.mu = np.array(mus)

        elif method == "default":
            # 随机选择k个样本作为初始均值向量
            self.mu = X[random.sample(range(X.shape[0]), self.k)]

        else:
            raise NotImplementedError

    def fit(self, X):
        self.init(X, "kmeans++")
        while True:
            C = {}
            for i in range(self.k):
                C[i] = []
            for j in range(X.shape[0]):
                # 距离
                d = np.sqrt(np.sum((X[j] - self.mu) ** 2, axis=1))
                # 簇标记
                lambda_j = np.argmin(d)
                # 将样本划入相应的簇
                C[int(lambda_j)].append(j)
            mu_ = np.zeros((self.k, X.shape[1]))
            for i in range(self.k):
                # 更新均值向量
                mu_[i] = np.mean(X[C[i]], axis=0)
            if np.sum((mu_ - self.mu) ** 2) < 1e-8:
                # 最小调整幅度阈值
                self.C = C
                break
            else:
                self.mu = mu_
        # 初始化标签
        self.labels_ = np.zeros((X.shape[0],), dtype=np.int32)
        for i in range(self.k):
            self.labels_[C[i]] = i

    def predict(self, X):
        preds = []
        for j in range(X.shape[0]):
            d = np.zeros((self.k,))
            for i in range(self.k):
                d[i] = np.sqrt(np.sum((X[j] - self.mu[i]) ** 2))
            preds.append(np.argmin(d))
        return np.array(preds)


if __name__ == "__main__":
    import pandas as pd
    import os

    # UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads.
    os.environ["OMP_NUM_THREADS"] = "1"

    # 读取西瓜数据集4.0
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../data/watermelon4.csv")
    data = pd.read_csv(file_path, header=None)
    X = data.values

    kmeans = KMeans(k=3)
    kmeans.fit(X)
    print(kmeans.C)
    print(kmeans.labels_)
    print(kmeans.predict(X))

    plt.subplots(1, 2, figsize=(12, 6))
    plt.subplot(121)
    # 坐标轴范围
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 0.8)
    # 坐标轴刻度
    plt.xticks([i / 10 for i in range(1, 10)])
    plt.yticks([i / 10 for i in range(9)])
    # 坐标轴名称
    plt.xlabel("density")
    plt.ylabel("sugar content")
    plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_)
    plt.scatter(kmeans.mu[:, 0], kmeans.mu[:, 1], c="red", marker="+")
    plt.title("kmeans")

    from sklearn.cluster import KMeans

    sklearn_kmeans = KMeans(n_clusters=3, n_init="auto")
    sklearn_kmeans.fit(X)
    print(sklearn_kmeans.labels_)
    plt.subplot(122)
    # 与第一个子图保持一致
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 0.8)
    plt.xticks([i / 10 for i in range(1, 10)])
    plt.yticks([i / 10 for i in range(9)])
    plt.xlabel("density")
    plt.ylabel("sugar content")
    plt.title("sklearn-kmeans")
    plt.scatter(X[:, 0], X[:, 1], c=sklearn_kmeans.labels_)

    plt.show()
