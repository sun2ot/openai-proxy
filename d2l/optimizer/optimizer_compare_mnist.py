import matplotlib.pyplot as plt
from d2l.mnist import load_mnist
from d2l.common.util import smooth_curve
from d2l.common.multi_layer_net import MultiLayerNet
from d2l.common.optimizer import *


# 读入MNIST数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 500


# 进行实验的设置
optimizers = {}
optimizers['SGD'] = SGD()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaGrad()
optimizers['Adam'] = Adam()
#optimizers['RMSprop'] = RMSprop()

networks = {}
train_loss = {}
for key in optimizers.keys():
    # 初始化四个神经网
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []    


# 开始训练
for i in range(max_iterations):
    # 6w选128
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in optimizers.keys():
        # 求梯度
        grads = networks[key].gradient(x_batch, t_batch)
        # 更新参数
        optimizers[key].update(networks[key].params, grads)
        # 统计损失函数
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    # 每迭代100次输出结果
    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))


# 绘制图形
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()
