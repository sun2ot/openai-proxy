import numpy as np

# 生成np数组
# (数组|暴露数组接口) -> ndarray(NumPy数组)
x = np.array([1., 2., 3.])
y = np.array([2., 4., 6.])
print(x, type(x))

# element-wise 对应元素计算
print('x+y={}'.format(x + y))
print('x-y={}'.format(x - y))
print('x*y={}'.format(x * y))
print('x/y={}'.format(x / y))

# n维数组
a = np.array([[1, 2], [3, 4]])
print('a为：\n', a)
print('a的形状：', a.shape)
print('a的数据类型：', a.dtype)

# 矩阵对应元素运算
b = np.array([[1, 9], [9, 7]])
print('a+b=\n', a + b)
print('a*b=\n', a * b)

# broadcast 广播(与标量进行运算)
x /= 2.
print(x, type(x))
c = np.array([10, 20])  # 与 a,b 形状不同
## 广播机制, 令 c=[10,20] -> [[10,20],[10,20]], 与 a 形状相同
print('a*c=\n', a * c)

# 访问元素
print('a的第一行', a[0])  # 0行
print('a的第一行第二个', a[0][1])  # 0行1列

# 将np数组压缩为一维
oda = a.flatten()
print('one dimension a:', oda)
print('索引为0,1的元素：', oda[np.array([0, 1])])
print('oda中大于2的元素', oda[oda > 2])  # 条件选择

# 多维数组
m = np.array([1, 2, 3, 4])
print('m的维数:', np.ndim(m))
print('m的形状:', m.shape)  # 返回 tuple

# 矩阵乘法
print(np.dot(a, b))  # 左行x右列




