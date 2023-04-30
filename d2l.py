import torch
import math
x = torch.tensor([[0,2,1],[3,0,2]],dtype=torch.float) # 2x3矩阵
y = torch.tensor(math.pow(44,1/3)) # 手算L3范数
z = torch.norm(x,p=3) # norm函数

print(y,z) # 
