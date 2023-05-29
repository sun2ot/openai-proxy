# 类的定义
class Man:
    # 属性
    name = '张三'
    age = 18
    # 私有属性，外部不可访问
    __id = '420102'

    # 构造方法,需传入自身 self
    def __init__(self, name, age, id):
        self.name = name
        self.age = age
        self.__id = id
        print('initialized!')

    # 普通方法
    def bark(self):
        print("I‘m " + self.name + "!")


# 实例化类
m = Man("Hogan", 22, '420102')
m.bark()
print(m.age)


class Test:
    def prt(haha):  # 通常用 self，真不用也行，但**必须要有**
        print(haha)  # haha 表示类的实例，即当前对象的地址
        print(haha.__class__)  # 指向类


t = Test()
t.prt()


# 继承
class Xiao(Man):
    address = ''

    # 不声明：继承父类构造方法
    # 声明：重写构造方法
    # 既要又要：用super调用父类构造方法，然后重写
    def __init__(self, address):
        self.address = address

    def getAge(self, age):
        self.age = age


x = Xiao('China')
x.getAge(99)
print(x.age)


# 类定义
class people:
    # 定义基本属性
    name = ''
    age = 0
    # 定义私有属性,私有属性在类外部无法直接进行访问
    __weight = 0

    # 定义构造方法
    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.__weight = w

    def speak(self):
        print("%s 说: 我 %d 岁。" % (self.name, self.age))


# 单继承示例
class student(people):
    grade = ''

    def __init__(self, n, a, w, g):
        # 调用父类的构函
        people.__init__(self, n, a, w)
        self.grade = g

    # 覆写父类的方法
    def speak(self):
        print("%s 说: 我 %d 岁了，我在读 %d 年级" % (self.name, self.age, self.grade))


# 另一个类，多继承之前的准备
class speaker():
    topic = ''
    name = ''

    def __init__(self, n, t):
        self.name = n
        self.topic = t

    def speak(self):
        print("我叫 %s，我是一个演说家，我演讲的主题是 %s" % (self.name, self.topic))


# 多继承
class sample(speaker, student):
    a = ''

    def __init__(self, n, a, w, g, t):
        student.__init__(self, n, a, w, g)
        speaker.__init__(self, n, t)


test = sample("Tim", 25, 80, 4, "Python")
test.speak()  # 方法名同，默认调用的是在括号中参数位置排前父类的方法
