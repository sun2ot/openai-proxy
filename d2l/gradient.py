import numpy as np
from NeuralNetwork import sigmoid,softmax

def function_2(x):
    return x[0] ** 2 + x[1] ** 2


def numerical_gradient(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # 生成和x形状相同的数组
    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)
        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # 还原值
    return grad


a = numerical_gradient(function_2, np.array([3.0, 4.0]))
print(a)


# 梯度下降
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

# initx1 = np.array([-3.0, 4.0])
# initx2 = np.array([-3.0, 4.0])
# lr1 = 10.
# lr2 = 1e-10
# print('lr too big: ', gradient_descent(function_2,initx1,lr1,100))
# print('lr too small: ', gradient_descent(function_2,initx2,lr2,100))

