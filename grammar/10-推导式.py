print('-' * 10 + '列表推导式' + '-' * 10)

# 列表推导式 = [out_exp_res for out_exp in input_list if condition]
names = ['Bob', 'Tom', 'alice', 'Jerry', 'Wendy', 'Smith']
# 若字符串长度>3，将其转化为大写
new_names = [name.upper() for name in names if len(name) > 3]
print(new_names)

# 计算 30 以内可以被 3 整除的整数
multiples = [i for i in range(30) if i % 3 == 0]
print(multiples)

print('-' * 10 + '字典推导式' + '-' * 10)

# 字典推导式 = { key_expr: value_expr for value in collection if condition }
listDemo = ['Google', 'Runoob', 'Taobao']
# 将列表中各字符串值为键，各字符串的长度为值，组成键值对
newDict = {key: len(key) for key in listDemo}  # 这里的key是for循环中的变量key，当然也可以换成别的
print(newDict)

# 提供三个数字，以三个数字为键，三个数字的平方为值来创建字典
dic = {x: x**2 for x in (2, 4, 6)}
print(dic)

print('-' * 10 + '集合推导式' + '-' * 10)

# 集合推导式 = { expression for item in Sequence if conditional }
# 计算数字 1,2,3 的平方数作为集合
setNew = {i ** 2 for i in (1, 2, 3)}
print(setNew)

# 判断不是 abc 的字母并输出
a = {x for x in 'abracadabra' if x not in 'abc'}
print(a)

print('-' * 10 + '元组推导式' + '-' * 10)

# 元组推导式 = (expression for item in Sequence if conditional )
b = (x for x in range(1, 10))
print(b)  # 元组推导式返回的结果是一个生成器对象
print(tuple(b))

print('-' * 10 + '推导式嵌套' + '-' * 10)

# 推导式嵌套

matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]

result = []

# debug
for i in range(4):
    ll = []
    for row in matrix:
        item = row[i]
        ll.append(item)
    result.append(ll)
# 这个列表推导式的结果如果看不懂，就去debug上面的for循环，结果和原理都是一样的
result2 = [[row[i] for row in matrix] for i in range(4)]

print(result, result2)
