import numpy as np
import matplotlib.pyplot as plt
from d2l.mnist import load_mnist
from d2l.common.multi_layer_net_extend import MultiLayerNetExtend
from d2l.common.optimizer import SGD, Adam

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 减少学习数据
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 10
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def __train(weight_init_std):
    # batch norm层
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    # 神经网络
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    # 优化器SGD
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
    
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            print("epoch:" + str(epoch_cnt) + " | 不使用batch norm的精确度: " + str(train_acc) + " | 使用: " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list


# 绘制图形
# 定义了一个长度为16的一维数组，其中的元素是对数空间中从10^0到10^-4之间均匀分布的数
weight_scale_list = np.logspace(0, -4, num=8)
x = np.arange(max_epochs)
# enumerate()将可迭代对象转变为枚举对象
for i, w in enumerate(weight_scale_list):
    print("============== " + str(i+1) + "/8" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    
    plt.subplot(4,2,i+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.title("W:" + str(w))
    if i == 8:
        # markevery：指定在曲线上绘制标记的间隔数量
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", label='Normal', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    # plt.legend(loc='lower right')
    
plt.show()