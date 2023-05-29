# 集合：无序、不重复的元素序列
eSet = {'item1', 'item2', 3, 4., ('5', '6')}  # {}创建
print(eSet, type(eSet))
emptySet = set([1, 2, 3, 4])  # set()创建
print(emptySet, type(emptySet))

print()

basket = {'apple', 'orange', 'apple', 'pear', 'orange', 'banana'}
print(basket)  # 集合元素不可重复，会自动去重

print()

# 集合运算
a = set('abracadabra')
b = set('alacazam')
print(a, b)  # 去重
print(a - b)  # a包含b不包含 (差集)
print(a | b)  # a或b包含 (并集)
print(a & b)  # a和b都包含 (交集)
print(a ^ b)  # 不同时包含于a和b

print()

# 添加元素，若元素已存在，则不进行任何操作
c = eSet.add('addItem')
print(eSet)

print()

# update()也可以添加元素
thisSet = {"Google", "Runoob", "Taobao"}
thisSet.update({1, 3})
print(thisSet)
thisSet.update([1, 4], [5, 6])
print(thisSet)

print()

# 移除元素
thisSet.remove('Google')
print(thisSet)
# thisSet.remove('noExistItem')  # 移除不存在的元素会报错

# discard()也能移除元素，且移除不存在的元素时不会报错
thisSet.discard('yahoo')
print(thisSet)

# pop()可以 随机删除 集合中的一个元素
thisSet.pop()
print(thisSet)  # 多执行几次，会发现不同的结果

