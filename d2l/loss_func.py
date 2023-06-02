import numpy as np
import mnist


# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# one-hot: 2为正确解
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# sigmoid结果: 2的概率最高
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

print('right_predict: ', mean_squared_error(np.array(y1), np.array(t)))
print('wrong_predict: ', mean_squared_error(np.array(y2), np.array(t)))


# 交叉熵误差
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


print('right_predict: ', cross_entropy_error(np.array(y1), np.array(t)))
print('wrong_predict: ', cross_entropy_error(np.array(y2), np.array(t)))

# 导入minis数据集
(x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize=True, one_hot_label=True)

# 训练数据样本数
train_size = x_train.shape[0]
# 选择样本数
batch_size = 10
# 随机选择
batch_index = np.random.choice(train_size, batch_size)
x_batch = x_train[batch_index]
t_batch = t_train[batch_index]


def cross_entropy_error_mini(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


