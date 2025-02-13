import numpy as np
import pickle
from tensorflow.keras.datasets import fashion_mnist

def save_fashion_mnist_partition(train_data, train_labels, test_data, test_labels, filename):
    """保存 Fashion-MNIST 数据为原始格式"""
    with open(filename, 'wb') as f:
        pickle.dump({'train_data': train_data, 
                     'train_labels': train_labels, 
                     'test_data': test_data, 
                     'test_labels': test_labels}, 
                    f, protocol=pickle.HIGHEST_PROTOCOL)

# 加载 Fashion-MNIST 数据集
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 获取类别数量
num_classes = 10  # Fashion-MNIST 有 10 个类别

# 创建训练集和测试集的类别索引
y_train = y_train.flatten()
y_test = y_test.flatten()
train_class_indices = {i: np.where(y_train == i)[0] for i in range(num_classes)}
test_class_indices = {i: np.where(y_test == i)[0] for i in range(num_classes)}

# 初始化划分后的索引
train1_indices, train2_indices = [], []
test1_indices, test2_indices = [], []

# 对每个类别进行均匀划分
for i in range(num_classes):
    # 训练集划分
    train_indices = train_class_indices[i]
    np.random.shuffle(train_indices)  # 随机打乱
    train_split = len(train_indices) // 2
    train1_indices.extend(train_indices[:train_split])
    train2_indices.extend(train_indices[train_split:])
    
    # 测试集划分
    test_indices = test_class_indices[i]
    np.random.shuffle(test_indices)  # 随机打乱
    test_split = len(test_indices) // 2
    test1_indices.extend(test_indices[:test_split])
    test2_indices.extend(test_indices[test_split:])

# 转换为 NumPy 数组
train1_indices = np.array(train1_indices)
train2_indices = np.array(train2_indices)
test1_indices = np.array(test1_indices)
test2_indices = np.array(test2_indices)

# 根据索引提取数据
x_train1, y_train1 = x_train[train1_indices], y_train[train1_indices]
x_train2, y_train2 = x_train[train2_indices], y_train[train2_indices]
x_test1, y_test1 = x_test[test1_indices], y_test[test1_indices]
x_test2, y_test2 = x_test[test2_indices], y_test[test2_indices]

# 保存划分后的数据
save_fashion_mnist_partition(x_train1, y_train1, x_test1, y_test1, 'fashion_mnist_partition1.pkl')
save_fashion_mnist_partition(x_train2, y_train2, x_test2, y_test2, 'fashion_mnist_partition2.pkl')

print("Fashion-MNIST 数据集已划分并保存为 'fashion_mnist_partition1.pkl' 和 'fashion_mnist_partition2.pkl'")
