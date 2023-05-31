import numpy as np

# 简单与门
def And(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    # python 中的三元表达式写法
    result = 1 if (x1 * w1 + x2 * w2) > theta else 0
    return result


for x1 in [0, 1]:
    for x2 in [0, 1]:
        print(And(x1, x2))  # 0 0 0 1


# 与门(加上偏置)
def And1(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    result = 1 if b + x*w > 0 else 0
    return result


# 与非门
def Nand(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 仅权重和偏置不同
    b = 0.7
    result = 1 if b + x*w > 0 else 0
    return result



