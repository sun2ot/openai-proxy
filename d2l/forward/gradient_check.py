import numpy as np
from d2l.mnist import load_mnist
from two_layer_net import TwoLayerNet
import time

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 选取前100个数据
x_batch = x_train[:100]  # (100, 784)
t_batch = t_train[:100]  # (100, 10)

gn_time_list = []
gb_time_list = []

# 数值微分法
start1 = time.time()
grad_numerical = network.numerical_gradient(x_batch, t_batch)
end1 = time.time()

gn_time_list.append(end1-start1)

# 误差反向传播法
start2 = time.time()
grad_backprop = network.gradient(x_batch, t_batch)
end2 = time.time()

gb_time_list.append(end2-start2)

# for key in grad_numerical.keys():
#     # 梯度数据差异的绝对值的平均值
#     diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
#     print(key + ":" + str(diff))

print(np.average(gn_time_list))
print(np.average(gb_time_list))

# 上述平均值求法实例
# a = np.array([1,2,3])
# b = np.array([4,5,6])
# print(c:=np.abs(a-b))
# print(np.average(c))