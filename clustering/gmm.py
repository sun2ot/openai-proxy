import numpy as np
import matplotlib.pyplot as plt


def multiGaussian(x, mu, sigma):
    # return (
    #     1
    #     / ((2 * np.pi) * pow(np.linalg.det(sigma), 0.5))
    #     * np.exp(-0.5 * (x - mu).dot(np.linalg.pinv(sigma)).dot((x - mu).T))
    # )
    return (
        1
        / ((2 * np.pi) * pow(np.linalg.det(sigma), 0.5))
        * np.exp(-0.5 * (x - mu).T.dot(np.linalg.pinv(sigma)).dot((x - mu)))
    )
    # todo: x-u的转置和x-u换个位置试试


def computeGamma(X, mu, sigma, alpha, multiGaussian):
    # 样本数
    n_samples = X.shape[0]
    # 簇数
    n_clusters = len(alpha)
    # 初始化 gamma 矩阵
    gamma = np.zeros((n_samples, n_clusters))

    p = np.zeros(n_clusters)

    g = np.zeros(n_clusters)

    for j in range(n_samples):
        for i in range(n_clusters):
            # 概率密度函数
            p[i] = multiGaussian(X[j], mu[i], sigma[i])
            # 高斯混合分布
            g[i] = alpha[i] * p[i]
        for k in range(n_clusters):
            gamma[j, k] = g[k] / np.sum(g)
    return gamma


class MyGMM:
    def __init__(self, n_clusters, ITER=50):
        """
        :param n_clusters: 高斯混合成分个数
        """
        self.n_clusters = n_clusters
        self.ITER = ITER
        self.mu = 0
        self.sigma = 0
        self.alpha = 0

    def fit(self, data):
        n_samples = data.shape[0]
        n_features = data.shape[1]

        # 初始化高斯混合分布的模型参数
        alpha = np.ones(self.n_clusters) / self.n_clusters
        # 为复现西瓜书效果，指定三个样本作为mu
        mu = np.array([[0.403, 0.237], [0.714, 0.346], [0.532, 0.472]])
        # mu=data[np.random.choice(range(n_samples), self.n_clusters)]
        # 协方差矩阵初始化为 0.1 的对角阵
        sigma = np.full(
            (self.n_clusters, n_features, n_features), np.diag(np.full(n_features, 0.1))
        )

        for t in range(self.ITER):
            gamma = computeGamma(data, mu, sigma, alpha, multiGaussian)
            alpha = np.sum(gamma, axis=0) / n_samples
            for i in range(self.n_clusters):
                mu[i] = (
                    # tips: 借助广播机制，对应元素相乘，不是矩阵乘法
                    np.sum(data * gamma[:, i].reshape((n_samples, 1)), axis=0)
                    / np.sum(gamma, axis=0)[i]
                )
                sigma[i] = 0
                for j in range(n_samples):
                    sigma[i] += (data[j].reshape((1, n_features)) - mu[i]).T.dot(
                        (data[j] - mu[i]).reshape((1, n_features))
                    ) * gamma[j, i]
                sigma[i] = sigma[i] / np.sum(gamma, axis=0)[i]
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha

    def predict(self, data):
        pred = computeGamma(data, self.mu, self.sigma, self.alpha, multiGaussian)
        cluster_results = np.argmax(pred, axis=1)
        return cluster_results


if __name__ == "__main__":
    import pandas as pd
    import os

    # 获取当前文件所在的目录路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建完整的文件路径
    file_path = os.path.join(current_dir, "../data/watermelon4.csv")
    data = pd.read_csv(file_path, header=None)
    data = data.values

    GMM = MyGMM(3)
    GMM.fit(data)
    result = GMM.predict(data)

    # 样本点
    plt.scatter(data[:, 0], data[:, 1], c=result)
    # 均值向量
    plt.scatter(GMM.mu[:, 0], GMM.mu[:, 1], marker="+", color="red")
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
    plt.title("GMM")
    plt.show()
