# ' 和 " 是等效的
var1 = 'hello world'
var2 = "wo cao"

# Python 不支持单字符类型，单字符在 Python 中也是作为一个字符串使用
char = 'c'
print(type(char))  # <class 'str'>

# 切片: 正向从0开始，逆向从-1开始，头索引必须小于尾索引，输出顺序是从左到右
print(var1[3:5])  # `lo`: var1的正向索引3~4位
print(var2[-4:-1])  # ` ca`: var2的逆向索引-4~-2位
print(var2 + ' ' + var1[6:])  # `wo cao world`: [6:]切片省略表示一直到头/尾
print(var1[:])  # `hello world`

# 转义字符(以下内容请在terminal中输入，否则可能出现与预期不符的情况)
print(
    'line1 \
    line2 \
    line3')  # 续航符
print('\\', '\'', '\"')
print('\a')  # 响铃
print('hello39\bworld')  # 退格
print('\n')  # 换行
print('hello \t world')  # 制表符
print('123456789\rabcdef')  # replace,将\r后的逐个替换到\r前的字符串: abcdef789

# 字符串运算符
a = 'tmd'
b = '小可爱'
print(a+b)  # 拼接
print(a*2)  # 重复2次
print(a[1])  # 字符串a的索引1
print(a[:3])
print('tm' in a)
print('wc' not in a)
print(r'这不是\n换行符')  # 字符串原义输出, r不区分大小写

# 字符串格式化
print('我叫%s，我%s还蛮大的' % ('阿杰', '房子'))  # 杰哥不要啊

# 三引号
# 三引号让程序员从引号和特殊字符串的泥潭里面解脱出来，自始至终保持一小块字符串的格式是所谓的WYSIWYG（所见即所得）格式的。
para_str = """这是一个多行字符串的实例
多行字符串可以使用制表符
TAB ( \t )。
也可以使用换行符 [二营长\n开炮]。
"""
print(para_str)

# f-string
cloud = '云'
water = '水'
str1 = f'{cloud}在青天{water}在瓶'
print(str1)

x = 1
print(f'{x+1}')   # Python 3.6
print(f'{x+1=}')   # Python 3.8: 可以使用 = 来拼接运算表达式与结果

# 字符串内建函数
dragon_man = 'beijing man'
print(dragon_man.capitalize())  # 首字符大写
print(dragon_man.center(20, '6'))  # 以字符串居中的 指定位数的 用指定字符填充的(默认空格)的 新字符串
print(dragon_man.count('i', 0, 4))  # 索引0-3中(左开右闭区间)，字符串i 出现的次数

print(dragon_man.endswith('man1'))  # 以指定字符串结尾
print(dragon_man.startswith('bei'))

print(dragon_man.find('jing'))  # 是否包含指定字符串，返回开始的索引值，没有返回-1
# print(dragon_man.index('666'))  # 作用和find()一样，但找不到会抛出异常

phone = 'huawei畅享'
vpn = '老王'
browser = '360'
proxy = 'Clash'
lion_awakens = '一九四九拾壹'
what_is_64 = '   '

print(phone.isalnum())  # is alphabet or Chinese character or number
print(phone.isalpha(), vpn.isalpha())  # is alphabet or Chinese character

print(browser.isdigit())  # only 阿拉伯数字
print(lion_awakens.isnumeric())  # all 数字(中文, 罗马, 全角, Unicode)
print(browser.isdecimal())  # 十进制数字

print(proxy.islower())  # only 小写
print(proxy.isupper())  # only 大写
print(proxy.lower())  # 大写转小写
print(proxy.upper())  # 小写转大写
print(proxy.swapcase())  # 大小写互换

print(what_is_64.isspace())  # only 空白

s1 = ('日', '你', '妈', '退', '钱')
sep = '-'
print(sep.join(s1))  # 将序列中的元素以指定的字符连接生成一个新的字符串(注意是谁调用谁)

print(len(phone))  # 字符串长度

print(phone.lstrip('huawei'))  # 截掉左边的字符/空格
print(phone.rstrip('畅享'))  # 截掉右边的字符/空格

print(max(phone), min(phone))  # 字符串中的最大/最小字符

print(phone.replace('畅享', '麦芒'))  # 替换

teachers = '凉森灵梦,三上悠亚,桥本有菜'
print(teachers.split(','))  # 以分隔符截取字符串





