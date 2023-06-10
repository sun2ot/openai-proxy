import numpy as np

from d2l.common.functions import sigmoid,softmax,cross_entropy_error

# 乘法层
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 前向传播 => x*y
    def forward(self, x, y):
        # 保存前向传播信号值
        self.x = x
        self.y = y                
        out = x * y

        return out

    # 反向传播 => 信号 * 翻转值
    def backward(self, dout):
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

# 加法层
class AddLayer:
    def __init__(self):
        pass

    # 前向 => x + y
    def forward(self, x, y):
        out = x + y

        return out

    # 后向 => 信号原样输出
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

class Relu:
    def __init__(self):
        # <=0 => True
        # >0 => False
        # mask为T/F组成的np数组
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        # mask为True的索引
        # 即x<=0的位置，对应out值置为0
        # 由此实现Relu函数的定义
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

# x = np.array( [[1.0, -0.5], [-2.0, 3.0]] )
# relu = Relu()
# print(relu.forward(x))


# sigmoid层
class Sigmoid:
    # 仅需要正向传播的输出值，即out
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        # 保存正向传播的输出
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


# 仿射变换层
class Affine:
    def __init__(self, W, b):
        """
        初始化仿射变换层
        :param W: 权重
        :param b: 偏置
        """
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 张量处理，不懂就看下面的实例
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        # 正向传播时，偏置会被加到每一个数据（第1个、第2个……）上
        # 反向传播时，偏置在输出通道的方向进行求和
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx

# 张量处理实例
# a = np.random.rand(2,3,4,5)
# print('a shape: ', a.shape)  # 2 3 4 5
# b = a.reshape(a.shape[0],-1)
# print('a reshape: ', b.shape)  # 2 60

# 偏置处理实例
# c = np.array([[1,2,3],[4,5,6]])
# c = np.sum(c, axis=0)
# print(c)  # [5 7 9]


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        # 交叉熵误差作为损失函数
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        """
        SoftmaxWithLoss反向传播的输出
        :param dout: 1
        :return: (y-t)/batch_size
        """
        batch_size = self.t.shape[0]
        # 监督数据是one-hot-vector的情况
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            # 非one-hot
            # 直接在输出值的对应索引位减1
            # 其余位假想减0，等价于不变
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class Dropout:
    """
    http://arxiv.org/abs/1207.0580
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # 生成的同型数组中，大于ratio的置为True，否则为False
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # 与x相乘后，为True的保留，为False的被置为0，达到drop节点的作用
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        # 反向传播时，行为与ReLU相同，即
        # 正向传播时传递了信号的神经元，反向传播时按原样传递信号
        # 正向传播时没有传递信号的，反向传播时信号将停在那里
        return dout * self.mask


class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv层的情况下为4维，全连接层的情况下为2维

        # 测试时使用的平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间数据
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            # var => \sigma^2
            var = np.mean(xc ** 2, axis=0)
            # std => \sqrt{\sigma^2 + \epsilon}
            std = np.sqrt(var + 10e-7)
            # xn => \hat{x_i}
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)

        dgamma = np.sum(self.xn * dout, axis=0)

        # dl/dx的第一部分
        dxn = self.gamma * dout
        dxc = dxn / self.std
        # dl/dx的第二部分（这写法真tm邪门，直接调math.pow不就好了)
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std

        # dl/dx的第三部分
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx