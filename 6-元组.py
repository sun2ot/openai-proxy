# 元组与列表相似，但其内的元素不能修改
tuple1 = ('雷军!', '金凡!')
tuple2 = ('字符串', 1001)  # 多类型
tuple3 = '闪电', '五连鞭'  # 也可以不加括号
tuple4 = ()  # 空元组
print(type(tuple1), type(tuple2), type(tuple3), type(tuple4))

print('-' * 20)

# 元组中只包含一个元素时，需要在元素后面添加逗号 , ，否则括号会被当作运算符使用
tupleWithNoComma = (50)
tupleWithComma = (50,)
print(type(tupleWithNoComma), type(tupleWithComma))

# 元组索引与列表相同，正向从0开始，负向从-1开始

print('-' * 20)

tuple5 = (12, 13, 14)
# tuple5[0] = 15  # 非法操作，元组值不可修改
tuple5 = (15, 16, 17)  # 元组不可修改指的是: 元组所指向的内存中的内容不可变
tuple6 = ('char1', 'char2')


del tuple5  # 删除元组

print('-' * 20)

# 元组运算符
print(len(tuple1))  # 2: 计算元组中元素个数
tuple_plus = tuple1 + tuple3  # 元组拼接
print(tuple_plus)
tuple6 += tuple3  # tuple6指向一个新元组
print(tuple6)

print(('焯',)*5)  # 复制

print(3 in (3, 4, 5))  # 元素是否存在










