import os
import numpy as np
import pickle
import cv2

data_dir = "./imagenet_ten"

IMG_SIZE = (32, 32)

categories = sorted(os.listdir(os.path.join(data_dir, "train")))
num_classes = len(categories)

def load_data(folder):
    images = []
    labels = []
    
    for class_idx, category in enumerate(categories):
        class_dir = os.path.join(folder, category)
        image_files = sorted(os.listdir(class_dir))
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            h, w, _ = img.shape
            if h < w:
                new_h, new_w = 256, int(w * (256 / h))
            else:
                new_h, new_w = int(h * (256 / w)), 256
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            start_x = (new_w - 224) // 2
            start_y = (new_h - 224) // 2
            img = img[start_y:start_y+224, start_x:start_x+224]
            
            images.append(img)
            labels.append(class_idx)
    
    return np.array(images, dtype=np.uint8), np.array(labels, dtype=np.int64).reshape(-1, 1)

x_train, y_train = load_data(os.path.join(data_dir, "train"))
x_test, y_test = load_data(os.path.join(data_dir, "test"))

train_class_indices = {i: np.where(y_train.flatten() == i)[0] for i in range(num_classes)}
test_class_indices = {i: np.where(y_test.flatten() == i)[0] for i in range(num_classes)}

train1_indices, train2_indices = [], []
test1_indices, test2_indices = [], []

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

train1_indices, train2_indices = np.array(train1_indices), np.array(train2_indices)
test1_indices, test2_indices = np.array(test1_indices), np.array(test2_indices)

x_train1, y_train1 = x_train[train1_indices], y_train[train1_indices]
x_train2, y_train2 = x_train[train2_indices], y_train[train2_indices]
x_test1, y_test1 = x_test[test1_indices], y_test[test1_indices]
x_test2, y_test2 = x_test[test2_indices], y_test[test2_indices]

x_train1 = x_train1.astype(np.uint8)
x_train2 = x_train2.astype(np.uint8)
x_test1 = x_test1.astype(np.uint8)
x_test2 = x_test2.astype(np.uint8)

y_train1 = y_train1.astype(np.uint8)
y_train2 = y_train2.astype(np.uint8)
y_test1 = y_test1.astype(np.uint8)
y_test2 = y_test2.astype(np.uint8)

def save_partition(filename, train_data, train_labels, test_data, test_labels):
    with open(filename, 'wb') as f:
        pickle.dump({'train_data': train_data, 
                     'train_labels': train_labels, 
                     'test_data': test_data, 
                     'test_labels': test_labels}, 
                    f, protocol=pickle.HIGHEST_PROTOCOL)

save_partition("imagenet10_partition1.pkl", x_train1, y_train1, x_test1, y_test1)
save_partition("imagenet10_partition2.pkl", x_train2, y_train2, x_test2, y_test2)

print("ImageNet-10 is divided into 'imagenet10_partition1.pkl' and 'imagenet10_partition2.pkl'")
