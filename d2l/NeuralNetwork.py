import numpy as np
import matplotlib.pylab as plt
import matplotlib


matplotlib.rc("font", family='MiSans')


# 阶跃函数
def OverStep(x):
    return (x > 0).astype(np.int32)


# sigmoid函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)

# softmax函数
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    return exp_a/sum_exp_a


# 优化的softmax函数
def opt_softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def Softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


# x = np.arange(-5.0, 5.0, 0.1)
# # y1 = OverStep(x)
# # y2 = sigmoid(x)
# y3 = relu(x)
# # plt.plot(x, y1, label="over-step")
# # plt.plot(x, y2, label="sigmoid")
# plt.plot(x, y3)
# # plt.title('阶跃函数与sigmoid函数')
# plt.title("ReLU函数")
# # plt.legend()
# plt.ylim(-1, 6)  # 制定y轴范围
# plt.show()

# 前向3层神经网络实现
def init_network():
    network = {}
    # 权值
    network['W1'] = np.array([[.1, .3, .5], [.2, .4, .6]])
    # 偏置
    network['b1'] = np.array([.1, .2, .3])
    network['W2'] = np.array([[.1, .4], [.2, .5], [.3, .6]])
    network['b2'] = np.array([.1, .2])
    network['W3'] = np.array([[.1, .3], [.2, .4]])
    network['b3'] = np.array([.1, .2])
    return network


def process(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 第一层的输出
    a1 = np.dot(x, W1) + b1
    # 激活函数
    z1 = sigmoid(a1)
    # 激活后的第一层输出作为第二层的输入
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = opt_softmax(a3)
    return y

# network = init_network()
# x = np.array([1., .5])
# print(y := process(network, x))


# a = np.array([0.3,2.9,4.0])
# print(b := opt_softmax(a), np.sum(b))
