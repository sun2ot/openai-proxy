# 字典名 = {key1: value1, key2: value2, ...}
# value可以是任何类型，但key只能是 数字/字符串
tinydict1 = {'abc': 456}
tinydict2 = {'abc': 123, 98.6: 37}

# 空字典
emptyDict = {}

# 打印字典
print(emptyDict)

# 查看字典的数量
print("Length:", len(emptyDict))

# 查看类型
print(type(emptyDict))

# 访问字典
print(tinydict1['abc'])

# 更新字典
tinydict2[98.6] = '阿巴阿巴'
print(tinydict2)

# 添加元素
tinydict2['add'] = 'sth.'
print(tinydict2)

# 删除字典元素
del tinydict2['add']
print(tinydict2)

# 清空字典
tinydict2.clear()
print(tinydict2)

# 删除字典
del tinydict2
# print(tinydict2)

# 字典中key唯一，若重复，则value以最后一次定义的为准
dupDict = {'name': '罪恶火花343', 'name': '偏见之僧'}
print(dupDict)

# 字典中key不可变，因此不能为列表，可以用数字，字符串或元组充当
exampleDict = {13: '14', 'str': '智库长', ('弗雷德-104', '凯利-087', '琳达-058'): 'Reach'}
print(exampleDict['弗雷德-104', '凯利-087', '琳达-058'])

# 字典内置函数
print(len(exampleDict))  # 字典元素个数
print(str(exampleDict))  # 字典的字符串表示
print(exampleDict.clear())  # 清空字典

