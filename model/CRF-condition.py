import math


def count_conditional_probability(w1, t, w2, s, x, y):

    """
    已知条件随机场模型，计算状态序列关于观测序列的条件概率

    :param w1: 转移特征权重
    :param t: 转移特征函数
    :param w2: 状态特征权重
    :param s: 状态特征函数
    :param x: 观测序列
    :param y: 状态序列
    :return: 状态序列关于观测序列的条件概率
    """

    # 转移特征数
    n_features_1 = len(w1)
    # 状态特征数
    n_features_2 = len(w2)
    # 序列中的位置数
    n_position = len(x)

    result = 0
    for k in range(n_features_1):
        for i in range(1, n_position):
            result += w1[k] * t[k](y[i - 1], y[i], x, i)
    for k in range(n_features_2):
        for i in range(n_position):
            result += w2[k] * s[k](y[i], x, i)
    return pow(math.e, result)

if __name__ == "__main__":

    def t1(y0, y1, x, i):
        return int(y0 in {0} and y1 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1)} and i in {1, 2})


    def t2(y0, y1, x, i):
        return int(y0 in {0} and y1 in {0} and x in {(1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)} and i in {1})


    def t3(y0, y1, x, i):
        return int(y0 in {1} and y1 in {0, 1} and x in {(0, 0, 0), (1, 1, 1)} and i in {2})


    def t4(y0, y1, x, i):
        return int(y0 in {1} and y1 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1),
                                                     (1, 0, 0), (1, 0, 1)} and i in {2})


    def t5(y0, y1, x, i):
        return int(y0 in {0, 1} and y1 in {0} and x in {(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)} and i in {1, 2})


    def s1(y0, x, i):
        return int(y0 in {0} and x in {(0, 1, 1), (1, 1, 0), (1, 0, 1)} and i in {0, 1, 2})


    def s2(y0, x, i):
        return int(y0 in {1} and x in {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0),
                                       (1, 0, 1)} and i in {0})


    def s3(y0, x, i):
        return int(y0 in {0} and x in {(0, 1, 1), (1, 1, 0), (1, 0, 1)} and i in {0, 1})


    def s4(y0, x, i):
        return int(y0 in {1} and x in {(1, 0, 1), (0, 1, 0)} and i in {0, 2})


    w1 = [1, 0.6, 1.2, 0.2, 1.4]
    t = [t1, t2, t3, t4, t5]
    w2 = [1, 0.2, 0.8, 0.5]
    s = [s1, s2, s3, s4]

    X = {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)}
    Y = {(0, 1, 0), (0, 1, 1), (0, 0, 0), (0, 0, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (1, 0, 1)}
    Z = 0

    result = []

    for x in X:
        for y in Y:
            r = [x, y, count_conditional_probability(w1, t, w2, s, x, y)]
            result.append(r)
            Z += r[2]
    for item in result:
        print(item[0], '->', item[1], ': ', item[2]/Z)