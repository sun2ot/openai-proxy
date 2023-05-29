# 一个自定义模块
# 注意模块名需要合法，否则 cus-module.py，这种就无法导入
def print_func(par):
    print("Hello : ", par)
    return


def it_is_a_very_long_name_func(str):
    print(str)
    return


def fun1():
    print('this is fun1')


def fun2():
    print('this is fun2')


if __name__ == '__main__':
    print('程序自身在运行')
else:
    print('我来自另一模块')


for name in dir():
    print(name)