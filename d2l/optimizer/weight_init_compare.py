import numpy as np
import matplotlib.pyplot as plt
from d2l.mnist import load_mnist
from d2l.common.util import smooth_curve
from d2l.common.multi_layer_net import MultiLayerNet
from d2l.common.optimizer import SGD


# 读入MNIST数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000


# 初始化权重字典（三种类型）
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
# 优化算法选择SGD
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    # 以三种不同的初始权重分别生成神经网络
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                  output_size=10, weight_init_std=weight_type)
    train_loss[key] = []


# 开始训练
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in weight_init_types.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.update(networks[key].params, grads)
    
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    
    # if i % 100 == 0:
    #     print("===========" + "iteration:" + str(i) + "===========")
    #     for key in weight_init_types.keys():
    #         loss = networks[key].loss(x_batch, t_batch)
    #         print(key + ":" + str(loss))


# 绘制图形
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()