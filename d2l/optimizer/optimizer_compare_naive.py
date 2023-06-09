import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from d2l.common.optimizer import *

def f(x, y):
    """
    定义函数f(x,y)
    """
    return x**2 / 20.0 + y**2

def df(x, y):
    """
    返回f(x,y)对x和对y的偏导
    :return: f'x, f'y
    """
    return x / 10.0, 2.0*y

# 起始点
init_pos = (-7.0, 2.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
# 初始梯度为0
grads = {}
grads['x'], grads['y'] = 0, 0


optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params['x'], params['y'] = init_pos[0], init_pos[1]

    # 优化器迭代30次
    for i in range(30):
        # 压入上次更新后的新参数值
        x_history.append(params['x'])
        y_history.append(params['y'])
        # 求梯度
        grads['x'], grads['y'] = df(params['x'], params['y'])
        # 用新的梯度更新参数
        optimizer.update(params, grads)
    

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)
    # 生成网格点坐标矩阵, shape=(1000,2000)
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
    
    # for simple contour line  
    mask = Z > 7
    Z[mask] = 0
    
    # 划分子图(2行2列，idx表示位置，可取值1-4)
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, 'o-', color="red")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    #colorbar()
    #spring()
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()