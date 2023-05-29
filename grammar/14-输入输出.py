# 将输出值转为字符串
import math

mainFrequency = 1.8
turboFrequency = 4.2

# str()  返回一个用户易读的表达形式。
# repr() 产生一个解释器易读的表达形式
print('主频为：' + repr(mainFrequency) + 'GHz,睿频为：' + str(turboFrequency) + 'GHz')

# repr() 函数可以转义字符串中的特殊字符
hello = "hello,world\n"
hellos = repr(hello)
print(hello, hellos)

# repr() 的参数可以是 Python 的任何对象
# 但注意只能有一个对象
test = repr((mainFrequency, turboFrequency, ('run away', 'aa')))
print(test)

# rjust(): 右对其，以指定字符填充左边空位(默认为空格)
# (字符串长度, 填充符)
for x in range(1, 10):
    print(repr(x).rjust(5, 'a'))  # aaaa1, aaaa2, ...

for x in range(1, 11):  # print()多项输出，自带空格分隔
    print(repr(x).rjust(2), repr(x*x).rjust(3), repr(x*x*x).rjust(4))


# str.format()
## {0:2d}: 字符串参数列表中的索引0位置，字符串长度为2，右对齐，空格填充
for x in range(1, 10):
    print('{0:2d} {1:3d}'.format(x, x*x))

## 对号入座
print('{}的网址是{}'.format("百度", "//baidu.com"))
## 括号中的数字对应参数列表的索引
print('{1}的外号叫{0}'.format('法外狂徒', '张三'))
## 指定名称(顺序无所谓)
print('{cloud}在青天{water}在瓶'.format(water='水', cloud='云'))
## 大杂烩(warn: 确保索引参数在关键字参数前面，否则会报错，例如`{car}xxx{1}`)
print('{0}年，我学会了开{car}'.format("1997", car = "汽车"))
## 格式控制
print('PI的近似值为{0:.3f}'.format(math.pi))


## zfill(): 返回指定位数字符串，右对齐，左边补0
print('123'.zfill(5))  # 00123

# 读取键盘输入
str1 = input("清输入\n")
print("你输入的是", str1)





