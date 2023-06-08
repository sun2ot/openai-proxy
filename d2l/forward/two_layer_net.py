from d2l.forward.layer import *
from d2l.common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        # OrderedDict为有序字典，可以记住向字典添加元素的顺序
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        # 输出层：用softmax正规化输出，交叉熵误差做损失函数
        self.lastLayer = SoftmaxWithLoss()
        
    def predict(self, x):
        # 该for循环迭代的是处理层的对象
        # 会按照affine1,relu1,affine2的顺序执行正向传播
        for layer in self.layers.values():
            x = layer.forward(x)
        # 返回输出数据
        return x
        
    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    # 精确度
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        # t为one-hot vector，转化为最大概率索引，即图片对应的数字
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy



    def numerical_gradient(self, x, t):
        """
        数值微分法计算权重、偏置的梯度
        :param x: 输入数据
        :param t: 监督数据
        :return: 权重、偏置的梯度字典
        """
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads


    def gradient(self, x, t):
        """
        误差反向传播法计算权重、偏置的梯度
        :param x: 输入数据
        :param t: 监督数据
        :return: 权重、偏置的梯度字典
        """
        # forward
        self.loss(x, t)

        # 输出层backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        # 获取所有中间层
        layers = list(self.layers.values())
        # 翻转字典以便逆序调用
        layers.reverse()
        # 逐层调用反向传播方法
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads

