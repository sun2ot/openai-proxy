print('-' * 10 + '创建迭代器' + '-' * 10)

# 迭代器只能往前不会后退
# 字符串，列表或元组对象都可用于创建迭代器
list1 = [1, 2, 3, 4]
it = iter(list1)  # 创建迭代器对象
print(next(it))  # 输出迭代器的下一个元素 1
print(next(it))  # 2

print('-' * 10 + '遍历迭代器' + '-' * 10)

# 迭代器对象可以使用常规for语句进行遍历
for x in it:  # 此时 x 从 3 开始
    print(x)
print()

# 也可以用 next() 遍历
import sys  # 引入 sys 模块

it2 = iter(list1)  # 创建一个新的迭代器对象


# while True:
#     try:
#         print(next(it2))
#     except StopIteration:
#         sys.exit()

# 如果不注释掉上面的部分，则下方的代码会被标识为 unreachable code
# 因为上面已经退出程序了

# 使用了 yield 的函数被称为生成器（generator）
# 生成器是一个返回迭代器的函数，只能用于迭代操作
# 在调用生成器运行的过程中，每次遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值, 并在下一次执行 next() 方法时从当前位置继续运行
def fibonacci(n):  # 生成器函数 - 斐波那契
    a, b, counter = 0, 1, 0
    while True:
        if counter > n:
            return
        yield a  # 执行到这里，保存并退出函数，然后返回yield值
        print('a=%d, b=%d, count=%d' % (a, b, counter))
        a, b = b, a + b
        counter += 1


# f 是一个迭代器，由生成器返回生成
# 此时没有执行fibonacci函数
f = fibonacci(10)

while True:
    try:
        # 每次next()拿到的都是yield返回的值
        # 此处开始执行fibonacci函数
        print(next(f), end=" ")
    except StopIteration:
        sys.exit()
