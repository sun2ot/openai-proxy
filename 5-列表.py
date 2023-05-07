# 列表
print('-' * 10, '索引访问', '-' * 10)
list1 = ['str1', 'str2', 1001., 88, "this is a list"]
for i in range(len(list1)):
    print(list1[i])  # 正向索引访问(0,1,2...)

print('-' * 20)

for j in range(-1, -len(list1), -1):
    print(list1[j])  # 逆向索引访问(-1,-2,-3,...)

print('-' * 10, '切片', '-' * 10)

print(list1[2:4])  # 切片(左闭右开区间)

print('-' * 10, '更新列表', '-' * 10)

# 更新列表
print(list1[1])
list1[1] = 114.514  # 修改
print(list1[1])
list1.append(True)  # 尾部挂载
print(list1)
del list1[0]
print(list1)  # 删除元素

print('-' * 10, '脚本操作符', '-' * 10)

# 脚本操作符
print(len(list1))  # 列表长度
print([1, 2, 3] + [4, 5, 6])  # 拼接
# print(list1 += ['盖亚!', 'dd'])  # 不能这么干哦
list1 += ['盖亚!']  # +=拼接，直接挂在后面
print(list1)
print(['阿巴'] * 4)  # 重复
print('只因' in ['只因', '尼', '太美'])  # 元素是否在列表中
for item in list1:
    print(item)  # 直接以元素迭代, 无需通过索引

print('-' * 10, '嵌套列表', '-' * 10)

# 嵌套列表
x = [['a', 'b', 'c'], [1, 2, 3]]  # 看成2x3的矩阵
print(x[0])
print(x[1][2])

print('-' * 10, '列表函数', '-' * 10)

# 列表函数
list2 = [1, 14, 51, 4, 4]

print(max(list2))  # 返回列表最大值: 注意，同类型数据才能比
print(min(list2))

t = (12, '3')
print(t, type(t))
print(y:=list(t), type(y))  # 将元组转位列表(还记得:=是什么运算符吗？看第二章)

print(list2.count(4))  # 统计元素出现次数

list1.extend(list2)  # 列表末尾扩展新的列表
print(list1)

print(list2.index(51))  # 返回第一个匹配项的索引

list2.insert(4, 9)  # 在第五个位置插入9
print(list2)

print(list2.pop())  # 弹出列表中一个元素并返回(默认参数为-1, 即最后一个)
print(list2)

list2.remove(51)  # 移除第一个匹配项
print(list2)

list2.reverse()  # 倒转列表中元素
print(list2)


