# 以 def 关键词开头，后接函数名和圆括号()
# return [表达式] 结束函数，选择性地返回一个值给调用方，不带表达式的 return 相当于返回 None
def hello():
    print("Hello World!")


result = hello()
print(result)


def max1(a, b):
    if a > b:
        return a
    else:
        return b


a = 4
b = 5
print(max1(a, b))

# 参数传递
# []和""分别是列表类型和字符串类型数据
# a是无类型的，仅仅是一个指针
a = [1, 2, 3]
a = "Runoob"

# 可更改(mutable)与不可更改(immutable)对象
# python 中一切都是对象，严格意义我们不能说值传递还是引用传递，我们应该说传不可变对象和传可变对象

print('-' * 10 + 'immutable' + '-' * 10)

'''
不可变类型：
变量赋值 a=5 后再赋值 a=10，
这里实际是新生成一个 int 值对象 10，再让 a 指向它，
而 5 被丢弃，不是改变 a 的值，相当于新生成了 a。
'''


def change(a):
    print('input id(a):', id(a))  # 指向的是同一个对象
    a = 10
    print('output id(a):', id(a))  # 一个新对象


a = 1
print('id(a):', id(a))
change(a)

print('-' * 10 + 'mutable' + '-' * 10)

'''
可变类型：
变量赋值 la=[1,2,3,4] 后再赋值 la[2]=5 
则是将 list la 的第三个元素值更改，
本身la没有动，只是其内部的一部分值被修改了
'''


def changeMe(myList):
    myList.append([1, 2, 3, 4])
    print("函数内取值: ", myList)
    return


mylist = [10, 20, 30]
changeMe(mylist)
print("函数外取值: ", mylist)

# 参数类型
print('-' * 10 + '关键字参数' + '-' * 10)


def printInfo(name, age):
    print("名字: ", name)
    print("年龄: ", age)
    return

# 调用函数时，指定参数名称，即关键字参数
# 关键字参数允许函数调用时参数的顺序与声明时不一致
printInfo(age=50, name="runoob")
# 关键字参数必须在普通参数的后面
# printInfo(name = '张三', 50)  # 报错

print('-' * 10 + '默认参数' + '-' * 10)


# 调用函数时，如果没有传递参数，则会使用默认参数
# 默认参数尽量使用不可变对象
def defaultNum(a=50):
    print(a)


defaultNum()

print('-' * 10 + '不定长参数' + '-' * 10)


# 加了星号 * 的参数会以元组(tuple)的形式导入，存放所有未命名的变量参数
def printinfo(arg1, *vartuple):
    print(arg1)
    print(vartuple)


printinfo(70, 60, 50)

print('-' * 20)


# 如果在函数调用时没有指定参数，它就是一个空元组。我们也可以不向函数传递未命名的变量
def printinfo(arg1, *vartuple):
    print(arg1)
    for var in vartuple:
        print(var)
    return


printinfo(10)
printinfo(70, 60, 50)

print('-' * 20)


# 加了两个星号 ** 的参数会以字典的形式导入
def printinfo(arg1, **vardict):
    print("**字典: ")
    print(arg1)
    print(vardict)


printinfo(1, a=2, b=3)

print('-' * 10 + '匿名函数 lambda' + '-' * 10)

# Python 使用 lambda 来创建匿名函数
# lambda [arg1 [,arg2,.....argn]]:expression
x = lambda a: a + 10
print(x(5))


# 可以将匿名函数封装在一个函数内，这样可以使用同样的代码来创建多个匿名函数
def myfunc(n):
    return lambda a: a * n


mydoubler = myfunc(2)
mytripler = myfunc(3)

print(mydoubler(11))
print(mytripler(11))


print('-' * 10 + '强制位置参数' + '-' * 10)

# 如果单独出现星号 *，则星号 * 后的参数必须用关键字传入
def f(a, b, *, c):
    print(a + b + c)


# f(1,2,3)  # 报错
f(1, 2, c=3)  # 正常

def f(a, b, /, c, d):
    print(a, b, c, d)


# /左边必须位置参数，右边可以位置，也可以关键字传入
f(10, 20, 30, d=40)


def f1(a, b, /, c, d, *, e, f):
    print(a, b, c, d, e, f)

# 同时出现了/和*，/左边必须位置，*右边必须关键字，中间部分两者都可
f1(10, 20, 30, d=40, e=50, f=60)


