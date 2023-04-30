a = 10
b = 3

# 算数运算符: 加 减 乘 除 取模 整除 幂(次方)
print(a + b, a - b, a * b, a / b, a % b, a // b, a**b)

# ---------------------------

# 比较运算符
print(a==b, a!=b, a>b, a<b, a>=b, a<=b)

# ---------------------------

# 赋值运算符: = += -= *= /= %= **= //= :=
# := 为海象运算符, 可在表达式内部为变量赋值. Python3.8 版本新增运算符
str1 = 'abcd'
print(c:=len(str1))

# ---------------------------

# 位运算符
a = 60  # 60 = 0011 1100
b = 13  # 13 = 0000 1101
c = 0

c = a & b  # 按位与
print("1 - c 的值为：", c)

c = a | b  # 按位或
print("2 - c 的值为：", c)

c = a ^ b  # 按位异或
print("3 - c 的值为：", c)

c = ~a  # 按位取反
print("4 - c 的值为：", c)

c = a << 2  # 左移2位: 1111 0000
print("5 - c 的值为：", c)

c = a >> 2  # 右移2位: 0000 1111
print("6 - c 的值为：", c)

# ---------------------------

# 逻辑运算符(boolean)
a = b = True
print(a and b)  # 布尔与: 如果 x 为 False, x and y 返回 x 的值, 否则返回 y 的计算值
print(1 and 233)
print(True or 999)  # 布尔或: 如果 x 是 True, 它返回 x 的值, 否则它返回 y 的计算值
print(not True)  # 布尔非: 如果 x 为 True, 返回 False. 如果 x 为 False, 它返回 True
print(not 3)  # False
print(not -1)  # False
print(not 0)  # True

# ---------------------------

# 成员运算符 in/not in
a = 10
b = 20
alist = [1, 2, 3, 4, 5]

if a in alist:
    print("1 - 变量 a 在给定的列表中 list 中")
else:
    print("1 - 变量 a 不在给定的列表中 list 中")

if b not in alist:
    print("2 - 变量 b 不在给定的列表中 list 中")
else:
    print("2 - 变量 b 在给定的列表中 list 中")

# 身份运算符: 判断两个标识符是不是引用自一个对象
a = 20
b = 20

if a is b:
    print("1 - a 和 b 有相同的标识")
else:
    print("1 - a 和 b 没有相同的标识")

if id(a) == id(b):  # id()用于获取对象内存地址
    print("2 - a 和 b 有相同的标识")
else:
    print("2 - a 和 b 没有相同的标识")

# 修改变量 b 的值
b = 30
if a is b:
    print("3 - a 和 b 有相同的标识")
else:
    print("3 - a 和 b 没有相同的标识")

if a is not b:
    print("4 - a 和 b 没有相同的标识")
else:
    print("4 - a 和 b 有相同的标识")

# is 和 == 的区别
a = [1,2,3]
b=a
print(b, b is a, b == a)
b=a[:]
print(b, b is a, b == a)
