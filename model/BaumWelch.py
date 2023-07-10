import numpy as np


def forward(A, B, PI, observation, only_r = True):
    """观测序列概率的前向算法"""
    n_state = len(A)  # 可能的状态数
    T = len(observation)
    alpha = np.zeros((T, n_state))

    # 初始化前向概率
    for i in range(n_state):
        alpha[0][i] = PI[i] * B[i][observation[0]]

    # 递推计算前向概率
    for t in range(1, T):
        for i in range(n_state):
            # alpha[t][i] = b[i, observation[t]] * np.sum(alpha[t - 1] * a[:, i])
            alpha[t][i] = np.dot(alpha[t-1], [a[i] for a in A]) * B[i][observation[t]]

    if not only_r: return alpha.tolist()

    return np.sum(alpha[len(alpha)-1])

def backward(A, B, PI, observation, only_r = True):
    """观测序列概率的后向算法"""
    n_state = len(A)  # 可能的状态数
    T = len(observation)

    # 初始化后向概率
    beta = np.ones((T, n_state))

    # 递推（状态转移）
    for t in range(len(observation) - 2, -1, -1):
        for i in range(n_state):
            beta[t][i] = np.dot(
                np.multiply(A[i], [b[observation[t+1]] for b in B]),
                beta[t+1]
            )

    if not only_r: return beta.tolist()

    return np.dot(
        np.multiply(PI, [b[observation[0]] for b in B]),
        beta[0]
    )

def baum_welch(observation, n_state, max_iter=100):
    """Baum-Welch算法学习隐马尔可夫模型

    :param observation: 观测序列
    :param n_state: 可能的状态数
    :param max_iter: 最大迭代次数
    :return: A,B,π
    """
    T = len(observation)  # 样本数
    n_observation = len(set(observation))  # 可能的观测数


    # ---------- 初始化随机模型参数 ----------
    # 初始化状态转移概率矩阵
    a = np.random.rand(n_state, n_state)
    a /= np.sum(a, axis=1, keepdims=True)

    # 初始化观测概率矩阵
    b = np.random.rand(n_state, n_observation)
    b /= np.sum(b, axis=1, keepdims=True)

    # 初始化初始状态概率向量
    p = np.random.rand(n_state)
    p /= np.sum(p)


    for _ in range(max_iter):
        # 计算前向概率
        alpha = forward(a,b,p,observation, only_r=False)
        # 计算后向概率
        beta = backward(a,b,p,observation, only_r=False)

        beta.reverse()

        # ---------- 计算：\gamma_t(i) ----------
        gamma = []
        for t in range(T):
            lst = np.zeros(n_state)
            for i in range(n_state):
                lst[i] = alpha[t][i] * beta[t][i]
            sum_ = np.sum(lst)
            lst /= sum_
            gamma.append(lst)

        # ---------- 计算：\xi_t(i,j) ----------
        xi = []
        for t in range(T - 1):
            sum_ = 0
            lst = [[0.0] * n_state for _ in range(n_state)]
            for i in range(n_state):
                for j in range(n_state):
                    lst[i][j] = alpha[t][i] * a[i][j] * b[j][observation[t + 1]] * beta[t + 1][j]
                    sum_ += lst[i][j]
            for i in range(n_state):
                for j in range(n_state):
                    lst[i][j] /= sum_
            xi.append(lst)

        # ---------- 计算新的状态转移概率矩阵 ----------
        new_a = [[0.0] * n_state for _ in range(n_state)]
        for i in range(n_state):
            for j in range(n_state):
                # 分子，分母
                numerator, denominator = 0, 0
                for t in range(T - 1):
                    numerator += xi[t][i][j]
                    denominator += gamma[t][i]
                new_a[i][j] = numerator / denominator

        # ---------- 计算新的观测概率矩阵 ----------
        new_b = [[0.0] * n_observation for _ in range(n_state)]
        for j in range(n_state):
            for k in range(n_observation):
                numerator, denominator = 0, 0
                for t in range(T):
                    if observation[t] == k:
                        numerator += gamma[t][j]
                    denominator += gamma[t][j]
                new_b[j][k] = numerator / denominator

        # ---------- 计算新的初始状态概率向量 ----------
        new_p = [1 / n_state] * n_state
        for i in range(n_state):
            new_p[i] = gamma[0][i]

        a, b, p = new_a, new_b, new_p
    return a, b, p

sequence = [0, 1, 1, 0, 0, 1]

A = [[0.0, 1.0, 0.0, 0.0], [0.4, 0.0, 0.6, 0.0], [0.0, 0.4, 0.0, 0.6], [0.0, 0.0, 0.5, 0.5]]
B = [[0.5, 0.5], [0.3, 0.7], [0.6, 0.4], [0.8, 0.2]]
PI = [0.25, 0.25, 0.25, 0.25]


alpha = forward(A, B, PI, sequence)
print("初始模型参数:", alpha)

A1,B1,P1 = baum_welch(sequence, 4, max_iter=1000)
alpha2 = forward(A1, B1, P1, sequence)
print("BN算法训练后:", alpha2)