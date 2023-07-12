import numpy as np


def bfgs_optimizer(objective_func, gradient_func, x0, tolerance=1e-6, max_iterations=100):
    n = len(x0)
    # 初始化Hessian逆矩阵为单位矩阵
    Bk = np.eye(n)
    xk = np.array(x0)
    gk = gradient_func(xk)
    k = 0

    while np.linalg.norm(gk) > tolerance and k < max_iterations:
        # 计算搜索方向
        pk = -np.dot(Bk, gk)
        # 通过线搜索确定步长
        _lambda = line_search(objective_func, gradient_func, xk, pk)
        xk1 = xk + _lambda * pk
        sk = xk1 - xk
        yk = gradient_func(xk1) - gk
        Bk = Bk + np.outer(yk, yk) / np.dot(yk, sk) - np.dot(np.dot(Bk, np.outer(sk, sk)), Bk) / np.dot(sk, np.dot(Bk, sk))
        xk = xk1
        gk = gradient_func(xk)
        k += 1

    return xk


def line_search(objective_func, gradient_func, xk, pk):
    _lambda = 1.0
    # 步长衰减因子
    rho = 0.9
    # 条件控制因子
    c = 0.0001
    while objective_func(xk + _lambda * pk) > objective_func(xk) + c * _lambda * np.dot(gradient_func(xk), pk):
        _lambda = rho * _lambda
    return _lambda


# 测试数据
def objective_func(x):
    return x[0] ** 2 + x[1] ** 2


def gradient_func(x):
    return np.array([2 * x[0], 2 * x[1]])


x0 = [1, 1]
result = bfgs_optimizer(objective_func, gradient_func, x0)
print("Optimized solution:", result)
