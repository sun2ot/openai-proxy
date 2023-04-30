import math

var1 = 1
var2 = 10
print(var1, var2)
# del 删除数字对象的引用
del var1, var2
# print(var1, var2)

# 整型int: Python3 整型是没有限制大小的，可以当作 Long 类型使用
num1 = 101
num2 = 10000111122223333
eight = 0o37  # 八进制
sixteen = 0xA0F  # 十六进制
print(type(num1), type(num2), type(eight), eight, type(sixteen), sixteen)

# 浮点型fload
num3 = 3.14
nun4 = 3.1415926535
num5 = 2.5e2  # 2.5*10^2
num6 = 2.  # 2.0
num7 = 3e+10  # e10 == e+10, e-10 == 10^(-10)
print(type(num3), type(nun4), type(num5), num5, type(num6), type(num7), num7)

# 复数complex: a + bj,或者complex(a,b)表示
#  复数的实部a和虚部b都是浮点型
com1 = 3.14j
com2 = 5 + 8j
com3 = 3e+26j  # (3*10^26)j
com4 = complex(6., 8)
com5 = .876j  # 0.876j
com6 = -.6545+0J  # -0.6545+0j
print(com1, com2, com3, com4, com5, com6)
com7 = complex(0o37, 0xA0F)
print(com7, type(com7))

# ----------------

# 数学函数
abs(-3)  # 绝对值
math.fabs(-3)  # 绝对值的浮点数形式

math.ceil(4.5)  # 向上取整
math.floor(4.5)  # 向下取整

# 比较cmp() python3已弃用
x=2
y=3
print((x>y), (x<y), (x>y)-(x<y))  # 替代cmp().  False=0,True=1

math.exp(3)  # 自然常数e的3次幂
math.log(3)  # log3(以e为底3的对数)
math.log10(3)  # log(10)3  以10为底3的对数

max(num1, num2, num3) # 返回最大值(参数类型需一致, 此处 num3 为 float)
min(num1, num2)  # 返回最小值

x = math.modf(3.14)  # 返回x的整数部分与小数部分，两部分的数值符号与x相同，整数部分以浮点型表示
print(x)

pow(3, 2)  # 3**2

print(round(3.31456, 3))  # 返回四舍五入后的值，保留小数点后3位
math.sqrt(9)  # 二次根号


