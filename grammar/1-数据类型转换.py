num_int = 123
num_flo = 1.23

num_new = num_int + num_flo

print("datatype of num_int:", type(num_int))
print("datatype of num_flo:", type(num_flo))

print("Value of num_new:", num_new)
print("datatype of num_new:", type(num_new))

# -----------------

num_int = 123
num_str = "456"

print("Data type of num_int:", type(num_int))
print("Data type of num_str:", type(num_str))

# 此时无法隐式转换
# print(num_int + num_str)
# 显示转换
print(num_int + int(num_str))

