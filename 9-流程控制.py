"""
if condition_1:
    statement_block_1
elif condition_2:        # python中的 else if 写作 elif
    statement_block_2
else:
    statement_block_3
"""
var1 = 100
if var1:  # if语句后加冒号: 引起子句
    print("1 - if 表达式条件为 true")  # 子句通过缩进构成语句块
    print(var1)
# 数值作为T/F判断是，非0为True，0为False
var2 = 0
if var2:
    print("2 - if 表达式条件为 true")
    print(var2)
print("Good bye!")

print()

# age = int(input("请输入你家狗狗的年龄: "))
# if age <= 0:
#     print("你是在逗我吧!")
# elif age == 1:
#     print("相当于 14 岁的人。")
# elif age == 2:
#     print("相当于 22 岁的人。")
# elif age > 2:
#     human = 22 + (age - 2) * 5
#     print("对应人类年龄: ", human)
# input("点击 enter 键退出")  # 退出提示

print()


# Python中没有switch...case，取而代之的是match...case
def http_error(status):
    match status:
        case 400:
            return "Bad request"
        case 404:
            return "Not found"
        case 418:
            return "I'm a teapot"
        case 502 | 500 | 403:
            return 'error'
        case _:  # 相当于C和Java中的default，兜底匹配
            return "Something's wrong with the internet"


myStatus = 400
print(http_error(400), http_error(500))

print('-' * 10 + 'while循环' + '-' * 10)

# while循环
n = 100
sum1 = 0
counter = 1
while counter <= n:
    sum1 = sum1 + counter
    counter += 1
print("1 到 %d 之和为: %d" % (n, sum1))

print('-' * 10 + 'while...else' + '-' * 10)

# python中while可以和else连用
count = 0
while count < 5:
    print(count, " 小于 5")
    count = count + 1
else:
    print(count, " 大于或等于 5")

# while循环体只有一条语句时，可以写在同一行
flag = 1
while flag < 3: print('flag = ', flag := flag + 1)

print('-' * 10 + 'for循环' + '-' * 10)

# for循环可以遍历任何 可迭代对象
sites = ["Baidu", "Google", "Runoob", "Taobao"]
for site in sites:  # 遍历列表
    print(site)
print()
word = 'runoob'  # 遍历字符串
for letter in word:
    print(letter)
print()
for number in range(1, 6):  # 遍历范围
    print(number)

print('-' * 10 + 'for...else' + '-' * 10)

# 在 Python 中，for...else 语句用于在循环结束后执行一段代码
for x in range(6):
    # 如果在循环过程中遇到了 break 语句，则会中断循环，此时不会执行 else 子句
    print(x)
else:
    print("Finally finished!")

# continue, break 用法没有变化，此处略

print('-' * 10 + 'pass语句' + '-' * 10)

# pass语句: 空语句，是为了保持程序结构的完整性
# 不做任何事情，一般用做占位语句
for letter in 'Runoob':
    if letter == 'o':
        pass
        print('执行 pass 块')
    print('当前字母 :', letter)



