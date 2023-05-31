import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font", family='MiSans')
# 定义角度范围
theta = np.linspace(0, 2 * np.pi, 1000)

# 计算心形线上的点 x 和 y 坐标
x = (1 + np.cos(theta)) * np.sin(theta)
y = (1 + np.cos(theta)) * np.cos(theta)

z = x - x

# 绘制心形线
plt.plot(x, y, label="heart")
plt.plot(x, z, linestyle="--", label="arrow")
# 设置标题
plt.title('心形线')
# 设置坐标轴标签
plt.xlabel('极地狐轴')
plt.ylabel('阿尼亚轴')
# 添加图例
plt.legend()
# 显示图形
plt.show()


# 读取图像
img = matplotlib.image.imread('./img/avatar.png')
plt.imshow(img)
plt.show()
