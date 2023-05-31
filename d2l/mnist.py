import pickle
import numpy as np
import os
from PIL import Image
from NeuralNetwork import process
import time

# 记录开始时间
start_time = time.time()

# 训练集6w张
train_num = 60000
# 测试集1w张
test_num = 10000
# 28x28像素
img_dim = (1, 28, 28)
# 28x28=784
img_size = 784

# 文件名字典
key_file = {
    'train_img': 'train-images.idx3-ubyte',
    'train_label': 'train-labels.idx1-ubyte',
    'test_img': 't10k-images.idx3-ubyte',
    'test_label': 't10k-labels.idx1-ubyte'
}


# 数据集路径
# os.path.dirname() 返回给定路径的目录名
# __file__返回当前脚本文件的绝对路径
dataset_dir = os.path.dirname(__file__) + '\\dataset'
save_file = dataset_dir + "\\mnist.pkl"


def _load_label(file_name):
    file_path = dataset_dir + "\\" + file_name

    print("Loading label" + file_name + " to NumPy Array ...")
    with open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("Done")

    return labels


def _load_img(file_name):
    file_path = dataset_dir + "\\" + file_name

    print("Loading img" + file_name + " to NumPy Array ...")
    with open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, img_size)
    print("Done")

    return data


def _convert2numpy():
    dataset = {}
    dataset['train_img'] = _load_img(key_file['train_img'])
    dataset['train_label'] = _load_label(key_file['train_label'])
    dataset['test_img'] = _load_img(key_file['test_img'])
    dataset['test_label'] = _load_label(key_file['test_label'])

    return dataset


# 序列化
def init_mnist():
    dataset = _convert2numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Done!")


def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T


def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    """读入MNIST数据集

    Parameters
    ----------
    normalize : 将图像的像素值正规化为0.0~1.0
    one_hot_label :
        one_hot_label为True的情况下，标签作为one-hot数组返回
        one-hot数组是指[0,0,1,0,0,0,0,0,0,0]这样的数组
    flatten : 是否将图像展开为一维数组

    Returns
    -------
    (训练图像, 训练标签), (测试图像, 测试标签)
    """
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0

    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])

    if not flatten:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def init_network():
    file_path = dataset_dir + "\\" + "sample_weight.pkl"
    with open(file_path, 'rb') as f:
        network = pickle.load(f)
    return network


# 显示数据集图片
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=False)
# print('x_train: ', x_train.shape)
# print('t_train: ', t_train.shape)
# img = x_train[0]
# label = t_train[0]
# # print('x_train[0]的label: ', label)
# print('img shape: ', img.shape)
# img_show(img)



# 分类测试
(x_train2, t_train2), (x_test2, t_test2) = load_mnist()
network = init_network()
accuracy_cnt = 0  # 精度
for i in range(len(x_test2)):
    y = process(network, x_test2[i])
    p = np.argmax(y)  # 获取概率最高的元素的索引(0,1,2...)
    if p == t_test2[i]:
        accuracy_cnt += 1  # 预测正确的图片数

print("Accuracy:" + str(float(accuracy_cnt) / len(x_test2)))

# 记录结束时间
end_time = time.time()
# 计算代码执行时间
duration = end_time - start_time
print("执行时间为：{:.2f}秒".format(duration))


# # 批处理
# batch_size = 100  # 批数量
# accuracy_cnt = 0
# for i in range(0, len(x_test2), batch_size):
#     x_batch = x_test2[i:i+batch_size]
#     y_batch = process(network, x_batch)
#     p = np.argmax(y_batch, axis=1)
#     accuracy_cnt += np.sum(p == t_test2[i:i+batch_size])
#
# print("Accuracy:" + str(float(accuracy_cnt) / len(x_test2)))
#
# # 记录结束时间
# end_time = time.time()
# # 计算代码执行时间
# duration = end_time - start_time
# print("批处理执行时间为：{:.2f}秒".format(duration))

