import numpy as np
from d2l.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 选取前3个数据
x_batch = x_train[:3]  # (3, 784)
t_batch = t_train[:3]  # (3, 10)


# # 数值微分法
# grad_numerical = network.numerical_gradient(x_batch, t_batch)
# # 误差反向传播法
# grad_backprop = network.gradient(x_batch, t_batch)
#
# for key in grad_numerical.keys():
#     # 梯度数据差异的绝对值的平均值
#     diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
#     print(key + ":" + str(diff))

# 上述平均值求法实例
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# print(c:=np.abs(a-b))
# print(np.average(c))