import numpy as np
import pickle
from tensorflow.keras.datasets import cifar100

def save_cifar100_partition(train_data, train_labels, test_data, test_labels, filename):
    with open(filename, 'wb') as f:
        pickle.dump({'train_data': train_data, 
                     'train_labels': train_labels, 
                     'test_data': test_data, 
                     'test_labels': test_labels}, 
                    f, protocol=pickle.HIGHEST_PROTOCOL)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

num_classes = 100

train_class_indices = {i: np.where(y_train.flatten() == i)[0] for i in range(num_classes)}
test_class_indices = {i: np.where(y_test.flatten() == i)[0] for i in range(num_classes)}

train1_indices = []
train2_indices = []
test1_indices = []
test2_indices = []

for i in range(num_classes):
    train_indices = train_class_indices[i]
    np.random.shuffle(train_indices)
    train_split = len(train_indices) // 2
    train1_indices.extend(train_indices[:train_split])
    train2_indices.extend(train_indices[train_split:])
    
    test_indices = test_class_indices[i]
    np.random.shuffle(test_indices)
    test_split = len(test_indices) // 2
    test1_indices.extend(test_indices[:test_split])
    test2_indices.extend(test_indices[test_split:])

train1_indices = np.array(train1_indices)
train2_indices = np.array(train2_indices)
test1_indices = np.array(test1_indices)
test2_indices = np.array(test2_indices)

x_train1 = x_train[train1_indices]
y_train1 = y_train[train1_indices]
x_train2 = x_train[train2_indices]
y_train2 = y_train[train2_indices]

x_test1 = x_test[test1_indices]
y_test1 = y_test[test1_indices]
x_test2 = x_test[test2_indices]
y_test2 = y_test[test2_indices]

x_train1 = x_train1.astype(np.uint8)
x_train2 = x_train2.astype(np.uint8)
x_test1 = x_test1.astype(np.uint8)
x_test2 = x_test2.astype(np.uint8)

y_train1 = y_train1.astype(np.uint8)
y_train2 = y_train2.astype(np.uint8)
y_test1 = y_test1.astype(np.uint8)
y_test2 = y_test2.astype(np.uint8)

save_cifar100_partition(x_train1, y_train1, x_test1, y_test1, 'cifar100_partition1.pkl')
save_cifar100_partition(x_train2, y_train2, x_test2, y_test2, 'cifar100_partition2.pkl')

print("CIFAR-100 is divided and saved as 'cifar100_partition1.pkl' and 'cifar100_partition2.pkl'")