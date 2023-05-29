# import module1[, module2[,... moduleN]
# 一个模块只会被导入一次，不管你执行了多少次 import
# 导入模块时，其内的主程序会自动在当前空间下运行，除非特殊标识
import support
import sys

print('-' * 10, '获取模块内名称', '-' * 10)
# support 模块下定义的所有名称
for name in dir(support):
    print(name)

support.print_func("Runoob")

print('-' * 10, '搜索路径', '-' * 10)
# python 模块的搜索路径存储在sys.path中
# 这是一个 list
# 在当前目录下存在与要引入模块同名的文件，就会把要引入的模块屏蔽掉
print(type(sys.path))
for item in sys.path:
    print(item)

print('-' * 10, '本地化函数', '-' * 10)
# 如果需要经常使用一个函数，可以把它赋给一个本地的名称
highFunc = support.it_is_a_very_long_name_func
highFunc('666')


print('-' * 10, 'from...import', '-' * 10)
# 导入一部分
from support import fun1, fun2
# from [module] import *  # 可以导入模块所有内容,由单一下划线（_）开头的名字不在此例
fun1()
fun2()
