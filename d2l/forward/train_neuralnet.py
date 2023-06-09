import numpy as np
from d2l.mnist import load_mnist
from two_layer_net import TwoLayerNet
import time

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
ng_time_list = []
g_time_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 6w个里选100个
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 数值差分求梯度
    start1 = time.time()
    grad0 = network.numerical_gradient(x_batch, t_batch)
    end1 = time.time()

    # 误差反向传播法求梯度
    start2 = time.time()
    grad = network.gradient(x_batch, t_batch)
    end2 = time.time()

    # 记录耗时
    ng_time_list.append(end1 - start1)
    g_time_list.append(end2 - start2)

    print('差分：{}，传播：{}'.format(np.average(ng_time_list), np.average(g_time_list)))

    # 更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad0[key]
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)

# print('差分：{}，传播：{}'.format(np.average(ng_time_list), np.average(g_time_list)))
# print('传播：{}'.format(np.average(g_time_list)))
# print('差分：{}'.format(np.average(ng_time_list)))