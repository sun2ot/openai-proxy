# coding: utf-8
from layer import *

apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# layer
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_fruit_layer = AddLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)   
orange_price = mul_orange_layer.forward(orange, orange_num)   
all_price = add_fruit_layer.forward(apple_price, orange_price)   
price = mul_tax_layer.forward(all_price, tax)   

# backward
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)   
dapple_price, dorange_price = add_fruit_layer.backward(dall_price)   
dorange, dorange_num = mul_orange_layer.backward(dorange_price)   
dapple, dapple_num = mul_apple_layer.backward(dapple_price)   

print("总价:", int(price))
print("苹果单价偏导:", dapple)
print("苹果数量偏导:", int(dapple_num))
print("橙子单价偏导:", dorange)
print("橙子数量偏导:", int(dorange_num))
print("税率偏导:", dtax)
