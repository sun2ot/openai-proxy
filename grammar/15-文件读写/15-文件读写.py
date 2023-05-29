# open(文件名, 模式)
f = open('test.txt', "w", encoding='utf-8')
## 返回写入字符数
num = f.write("666\n114514\n测试")
print('写入了{}个字符'.format(num))
f.close()

f = open('test.txt', "r", encoding='utf-8')
# str1 = f.read()
# print(str1)
# 上下两个read显然不能一起用，否则光标已经到文件末尾了，下一个啥也读不到
str2 = f.readlines()
print(str2)
f.close()

# 通过迭代对象逐行读取
f = open('test.txt', "r", encoding='utf-8')
for line in f:
    print(line, end='')
f.close()

