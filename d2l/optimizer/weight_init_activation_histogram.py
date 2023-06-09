import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)

# 随机生成一个元素符合标准正态分布（高斯分布）的数组作为输入层
input_data = np.random.randn(1000, 100)  # 1000个数据
node_num = 100  # 各隐藏层的节点（神经元）数
hidden_layer_size = 5  # 隐藏层有5层
activations = {}  # 激活值的结果保存在这里

x = input_data

for i in range(hidden_layer_size):
    # 如果不是第一层，将前一层输出作为本层输入
    if i != 0:
        x = activations[i-1]

    # 重新生成一个100x100的高斯分布数组作为本层权重
    #w = np.random.randn(node_num, node_num) * 1
    #w = np.random.randn(node_num, node_num) * 0.01

    # Xavier初始值
    #w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    # He初始值
    w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(x, w)


    # 激活后获得本层输出
    #z = sigmoid(a)
    z = ReLU(a)
    #z = tanh(a)

    # 保存本层输出，以便下一层使用
    activations[i] = z

for i, a in activations.items():
    # 按层数划分子图
    plt.subplot(1, len(activations), i+1)
    plt.subplots_adjust(wspace=0.5, hspace=0.3)
    plt.title(str(i+1) + "-layer")
    # 设置y轴刻度位置和标签
    plt.yticks(np.arange(0, 501, 100), np.arange(0, 501, 100))
    # 非输入层，隐藏y轴上的刻度标签和刻度线
    if i != 0:
        plt.yticks([], [])
    plt.xlim(0,1)
    plt.ylim(0, 500)

    # 绘制直方图：a扁平化，直方图30条，统计(0,1)间的数据
    plt.hist(a.flatten(), 100, range=(0,1))
plt.show()
